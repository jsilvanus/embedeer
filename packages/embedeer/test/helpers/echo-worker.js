/**
 * A minimal echo worker used only in tests.
 *
 * init  → sends { type: 'ready' }
 * task  → echoes each text as an embedding of [42]
 */
process.on('message', (msg) => {
  if (msg.type === 'init') {
    process.send({ type: 'ready' });
  } else if (msg.type === 'task') {
    process.send({
      type: 'result',
      id: msg.id,
      embeddings: msg.texts.map(() => [42]),
    });
  }
});
