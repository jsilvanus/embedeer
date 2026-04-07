/**
 * A worker that crashes immediately after receiving the init message.
 * Used in tests to verify that the parent process is not affected.
 */
process.on('message', (msg) => {
  if (msg.type === 'init') {
    process.exit(1);
  }
});
