# Photo Pre-processing in Agentic Loop

I want to add the following set of image pre-processing functions as an additional MCP tool to be executed before the first OCR step:


```py
import cv2
import numpy as np


def order_points(points):
    """Order four corners: top-left, top-right,
    bottom-right, bottom-left."""

    points = np.asarray(points, dtype="float32")

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()

    return np.array([
        points[np.argmin(s)],
        points[np.argmin(diff)],
        points[np.argmax(s)],
        points[np.argmax(diff)]
    ], dtype="float32")


def find_document_corners(image):
    """Find the largest four-corner document contour."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )

    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours[:15]:

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:
            area = cv2.contourArea(approx)

            if area > image.shape[0] * image.shape[1] * 0.10:
                return order_points(approx.reshape(4, 2))

    return None


def perspective_correct(image, corners=None):
    """Correct perspective distortion."""

    if corners is None:
        corners = find_document_corners(image)

    if corners is None:
        return image

    tl, tr, br, bl = order_points(corners)

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    width = int(max(width_a, width_b))
    height = int(max(height_a, height_b))

    if width < 2 or height < 2:
        return image

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        np.asarray(corners, dtype="float32"),
        destination
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def remove_shadows(gray):
    """
    Estimate uneven background illumination and
    normalize the document surface.
    """

    h, w = gray.shape

    # Estimate background at a much larger scale
    kernel_size = max(31, (min(h, w) // 20) | 1)

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=kernel_size / 3,
        sigmaY=kernel_size / 3
    )

    background = np.maximum(background, 1)

    normalized = (
        gray.astype(np.float32) /
        background.astype(np.float32)
    ) * 245.0

    return np.clip(normalized, 0, 255).astype(np.uint8)


def enhance_text(gray, sharpen=True):
    """Improve local contrast and text definition."""

    # Edge-preserving noise reduction
    denoised = cv2.bilateralFilter(
        gray,
        d=7,
        sigmaColor=35,
        sigmaSpace=35
    )

    # Local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    if sharpen:
        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            sigmaX=1.0
        )

        enhanced = cv2.addWeighted(
            enhanced, 1.5,
            blurred, -0.5,
            0
        )

    return enhanced


def preprocess_document(
    image,
    corners=None,
    output_size=None,
    create_binary=True,
    remove_shadow=True,
    sharpen=True
):
    """
    Complete document image preprocessing pipeline.

    Parameters
    ----------
    image:
        Image path or OpenCV BGR image array.

    corners:
        Optional manually supplied document corners.
        Order is arbitrary; they are sorted automatically.

    output_size:
        Optional (width, height) output dimensions.

    create_binary:
        Generate an adaptive thresholded image.

    remove_shadow:
        Enable illumination normalization.

    sharpen:
        Enable moderate sharpening.

    Returns
    -------
    dict:
        {
            "gray": enhanced grayscale image,
            "binary": adaptive threshold image,
            "aligned": perspective-corrected BGR image
        }
    """

    # Load image
    if isinstance(image, str):
        image = cv2.imread(image)

        if image is None:
            raise ValueError("Could not load input image.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be an image path or NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim == 2:
        image = cv2.cvtColor(
            image, cv2.COLOR_GRAY2BGR
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or BGR image.")

    # 1. Geometric correction
    aligned = perspective_correct(image, corners)

    if output_size is not None:
        aligned = cv2.resize(
            aligned,
            output_size,
            interpolation=cv2.INTER_CUBIC
        )

    # 2. Grayscale conversion
    gray = cv2.cvtColor(
        aligned,
        cv2.COLOR_BGR2GRAY
    )

    # 3. Illumination correction
    if remove_shadow:
        gray = remove_shadows(gray)

    # 4. Text enhancement
    enhanced = enhance_text(
        gray,
        sharpen=sharpen
    )

    return {
        "gray": enhanced
    }
```


I would implement this as a bounded, agent-controlled document preprocessing tool, using FastMCP and OpenCV.

The key architectural decision is that the agent should not be allowed to repeatedly call arbitrary OpenCV functions or modify every parameter. Instead, expose a small set of explicitly tunable parameters, each with:

* A default value.

* A permitted minimum and maximum.

* A step size or predefined choices.

* An explanation of what changing it does.

* Guidance about which image conditions justify changing it.

* A maximum number of attempts.

The agent should also receive structured feedback from OCR so that it can make an informed decision about whether another preprocessing attempt is worthwhile.

I would separate this into three components:

* Preprocessing engine: Performs the OpenCV operations.

* OCR evaluation engine: Determines whether extraction succeeded.

* FastMCP tool: Manages defaults, permitted adjustments, retries, and execution limits.

This is preferable to letting an LLM directly control an unrestricted preprocessing loop.



1. Proposed architecture

Agent ->  Requests invoice extraction -> FastMCP: extract_invoice (Enforces attempt budget and parameter schema) -> Default preprocessing: OpenCV with approved defaults -> OCR + extraction validation (CER, confidence, required fields, validation rules) -> Retry decision: Success / retry justified / unrecoverable failure

=> Success: Return extracted data
=> Retry: Agent selects permitted changes


I would use an important rule:

The agent recommends parameter changes; the server enforces whether they are valid and whether another attempt is permitted.

The agent must never control the retry counter.
2. Which parameters should the agent be allowed to change?

I would expose six parameters, grouped by the image problem they address.

These are deliberately selected because they are commonly adjusted when tuning document-scanning and OCR pipelines.


| Parameter | Default | Allowed range | When the agent should change it |
| -- | -- | -- | -- |
| Canny low threshold | 50 | 30–80 | Weak document edges or noisy backgrounds |
| Canny high threshold | 150 | 100–220 | Document boundaries are not being detected |
| CLAHE clip limit | 2.0 | 1.0–4.0 | Faint or low-contrast printing |
| CLAHE tile size | 8 | 4–16 | Local illumination varies across the document |
| Sharpening strength | 0.5 | 0–0.8 | Text appears slightly soft |

A few important details:

* Canny thresholds affect document boundary detection, not the actual OCR recognition.

* CLAHE parameters affect local contrast.

* Sharpening should only be increased when there is mild softness. It should not be used as a substitute for deblurring.

The parameters should not all be changed simultaneously. The agent should select changes based on the observed failure.



Parameter guidance exposed to the agent

I would actually make this guidance part of the MCP tool description or a separate resource.

For example:

Canny low threshold
Default: 50

Range: 30–80

Used to identify weaker edges during document boundary detection.

* Decrease toward 30–40 if document boundaries are faint.

* Increase toward 60–80 if background texture creates false edges.

CLAHE clip limit
Default: 2.0

Range: 1.0–4.0

Controls the degree of local contrast enhancement.

* Increase toward 3–4 for faded printing.

* Decrease toward 1–1.5 if noise and paper texture become prominent.

Adaptive threshold C
Default: 10

Range: 5–15

Controls how aggressively pixels are classified as foreground text.

* Increase if background artifacts are being classified as text.

* Decrease if faint characters are disappearing.

Adaptive threshold block size
Default: 31

Range: 15–51

Controls the local neighborhood used for threshold calculation.

* Decrease for small text and rapidly changing illumination.

* Increase for larger text and gradual illumination changes.

These explanations give the agent a basis for making decisions instead of encouraging random parameter experimentation.

3. Define what constitutes a failure

This is arguably the most important part of the design.

I would not define failure as simply "OCR confidence below 90%."

OCR confidence alone is insufficient: an OCR engine can confidently misread an invoice number or total.

Instead, use a combination of deterministic quality checks.

My suggested initial rules:

| Condition | Failure? |
| -- | -- |
| No document boundary found | Yes |
| OCR returns no text | Yes |
| Fewer than x recognized text lines | Yes, for a normal invoice |
| Average OCR confidence below 0.75 | Yes |
| Required field - e.g. invoice number missing | Yes |
| Required total missing | Yes, if configured |
| Arithmetic validation fails | Flag for review, not automatically an image-processing failure |


I would separate the outcomes into three categories:

* SUCCESS: Extraction meets the configured requirements.

* RETRYABLE: Extraction failed in a way that another preprocessing configuration might improve.

* MANUAL_REVIEW: The document is unreadable, required information is ambiguous, or extraction failed for reasons that image enhancement cannot reasonably fix.

For example, an arithmetic inconsistency is not necessarily a preprocessing problem. Retrying CLAHE five times will not fix a genuinely inconsistent invoice.


4. FastMCP implementation

I recommend exposing two tools rather than one tool that performs an uncontrolled loop.

* extract_invoice: Runs the default configuration and returns the OCR result and retry recommendations.

* retry_invoice: Accepts an attempt ID and a limited set of parameter changes, validates them, and runs another attempt.

This gives the agent control over which adjustments to make, while the server retains control over how many attempts can happen.

Recommended policy Maximum 4 attempts per document, including the initial attempt.


A. Parameter configuration and validation


```py

from dataclasses import dataclass, asdict
from typing import Optional
from pydantic import BaseModel, Field, model_validator


@dataclass
class PreprocessConfig:
* canny_low: int = 50
    canny_high: int = 150
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    threshold_block: int = 31
    threshold_c: int = 10
    sharpen_strength: float = 0.5


class RetryChanges(BaseModel):
    """Only explicitly approved parameters are exposed."""

    canny_low: Optional[int] = Field(
        None, ge=30, le=80,
        description=(
            "Lower threshold for document edge detection. "
            "Decrease for weak edges; increase for noisy backgrounds."
        )
    )

    canny_high: Optional[int] = Field(
        None, ge=100, le=220,
        description=(
            "Upper threshold for document edge detection. "
            "Adjust when document contours are not detected reliably."
        )
    )

    clahe_clip: Optional[float] = Field(
        None, ge=1.0, le=4.0,
        description=(
            "Local contrast enhancement strength. "
            "Increase for faded text; decrease if noise is amplified."
        )
    )

    clahe_tile: Optional[int] = Field(
        None, ge=4, le=16,
        description=(
            "CLAHE tile dimension. Use smaller tiles for localized "
            "illumination problems and larger tiles for gradual variation."
        )
    )

    threshold_block: Optional[int] = Field(
        None, ge=15, le=51,
        description=(
            "Adaptive threshold neighborhood size. Must be odd. "
            "Smaller values respond to local changes; larger values "
            "respond to broader illumination variations."
        )
    )

    threshold_c: Optional[int] = Field(
        None, ge=5, le=15,
        description=(
            "Adaptive threshold offset. Increase to suppress "
            "background artifacts; decrease to retain faint characters."
        )
    )

    sharpen_strength: Optional[float] = Field(
        None, ge=0.0, le=0.8,
        description=(
            "Unsharp-mask strength. Increase only for mildly soft text. "
            "Aggressive sharpening can create false character edges."
        )
    )

    @model_validator(mode="after")
    def validate_combinations(self):
        if self.threshold_block is not None:
            if self.threshold_block % 2 == 0:
                raise ValueError(
                    "threshold_block must be an odd number."
                )

        if (
            self.canny_low is not None
            and self.canny_high is not None
            and self.canny_low >= self.canny_high
        ):
            raise ValueError(
                "canny_low must be lower than canny_high."
            )

        return self
```

An additional server-side check is important: even if only one Canny threshold is supplied, the resulting combination must still satisfy low < high.


B. Update the OpenCV preprocessing functions

The previous preprocessing implementation can be retained with a few changes.

The important part is passing the configuration object down into the individual operations.


```py

import cv2
import numpy as np


def find_document_corners(image, config):

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    edges = cv2.Canny(
        blurred,
        config.canny_low,
        config.canny_high
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )

    edges = cv2.dilate(
        edges, kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours[:15]:

        perimeter = cv2.arcLength(
            contour, True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:
            return order_points(
                approx.reshape(4, 2)
            )

    return None


def enhance_text(gray, config):

    denoised = cv2.bilateralFilter(
        gray, 7, 35, 35
    )

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip,
        tileGridSize=(
            config.clahe_tile,
            config.clahe_tile
        )
    )

    enhanced = clahe.apply(denoised)

    if config.sharpen_strength > 0:

        blurred = cv2.GaussianBlur(
            enhanced, (0, 0), 1.0
        )

        strength = config.sharpen_strength

        enhanced = cv2.addWeighted(
            enhanced,
            1.0 + strength,
            blurred,
            -strength,
            0
        )

    return enhanced


def preprocess_with_config(image, config):

    corners = find_document_corners(
        image, config
    )

    aligned = (
        perspective_correct(image, corners)
        if corners is not None
        else image.copy()
    )

    gray = cv2.cvtColor(
        aligned, cv2.COLOR_BGR2GRAY
    )

    gray = remove_shadows(gray)

    enhanced = enhance_text(
        gray, config
    )

    return {
        "gray": enhanced,
        "document_detected": corners is not None
    }

```


Here, order_points, perspective_correct, and remove_shadows are the helper functions from the previous implementation.


C. FastMCP server with enforced retry limits

This is where the important orchestration happens.

The server maintains:

* Original image.

* Current preprocessing configuration.

* Attempt count.

* Previous extraction results.

* Previously attempted configurations.

* Retry eligibility.

The agent only receives an attempt_id; it cannot reset the counter by submitting another retry request.

A few production changes are also necessary:

* Add expiration and cleanup for abandoned sessions.

* Store sessions in Redis or a database if multiple workers are used.

* Add an upload mechanism or a controlled document-storage connector rather than accepting arbitrary filesystem paths.

* Authenticate users and ensure an attempt ID is only accessible to its owner.

* Put limits on image dimensions, decoded file size, and processing time.


D. Track improvement, not just success or failure

Store the quality metrics for every attempt.

For example:

| Attempt | Confidence | Recognized lines | Missing fields |
| -- | -- | -- | -- |
| 1 | 0.61 | 8 | 2 |
| 2 | 0.72 | 12 | 1 |
| 3 | 0.89 | 18 | 0 |

These + timestamp should be added to the SQL entry so that the agent is able to execute, e.g. "Show me invoices that had errors on {date}" or "Show me invoices where the confidence was below {threshold}"


E. Preserve the best result

Always keep the original image and never overwrite the best extraction with the latest attempt.

Keep the highest-quality result, including its configuration (update in SQLite when improved) and processed image by appending "_processed" to it's filename. The agent should be able to retrieve both file path on a users request.


F. Expose history

For implementation, I would also consider exposing a third read-only MCP tool:

get_preprocessing_attempt_history(attempt_id)

This would allow the agent to inspect previous configurations, OCR confidence, missing fields, and processing outcomes before choosing its next adjustment.
