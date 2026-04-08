module.exports = [
  {
    files: ['**/*.{js,mjs,cjs}'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        performance: 'readonly',
      },
    },
    plugins: {},
    rules: {
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
      'no-console': 'off',
    },
  },
  {
    files: ['test/**'],
    rules: {
      'no-unused-vars': 'off',
    },
  },
];
