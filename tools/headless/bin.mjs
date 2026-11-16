#!/usr/bin/env node
// The `shaderous` command of the npm package: `shaderous mcp [options]` starts the MCP server
// (mcp.mjs); anything else goes to the command-line runner (cli.mjs). So one package name works
// with npx for both: `npx -y shaderous mcp`, `npx -y shaderous run "problem=sedov"`.
if (process.argv[2] === 'mcp') {
  process.argv.splice(2, 1);
  await import('./mcp.mjs');
} else {
  await import('./cli.mjs');
}
