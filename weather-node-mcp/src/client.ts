import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';

// Let the client start the server
const client = new Client({ name: 'my-client-id', version: '1.0.0' });

const transport = new StdioClientTransport({
    command: 'npx',
    args: ['tsx', 'src/index.ts']
});

await client.connect(transport);


// Call server resources
const { resources } = await client.listResources();
console.log(resources);

const { contents } = await client.readResource({ uri: 'weather://about' });
console.log(contents);


// List all tools
const { tools } = await client.listTools();
for (const tool of tools) {
    console.log(tool.name, '—', tool.description);
}


// Calling a tool
const result = await client.callTool({ name: 'get-alerts', arguments: { state: 'AZ' } });

for (const block of result.content) {
    if (block.type === 'text') console.log(block.text);
}

await client.close();
