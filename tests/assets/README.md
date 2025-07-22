# Test Assets

This directory contains audio files used for integration testing.

## Audio Files

### Guidelines

- Keep files small (< 1MB each) to avoid bloating the repository
- Use short duration samples (1-30 seconds)
- Include various formats (MP3, WAV, FLAC) to test format support
- Document the source and content of each file

### Recommended Test Files

1. **short_sample.wav** (1-2 seconds)
   - Simple spoken phrase like "Hello world"
   - Uncompressed format for baseline testing

2. **medium_sample.mp3** (5-10 seconds)
   - Short sentence or phrase
   - Compressed format testing

3. **long_sample.flac** (20-30 seconds)
   - Longer speech sample for performance testing
   - Lossless compressed format

### Sources

- Use royalty-free/public domain audio
- Consider generating synthetic speech with TTS
- Use samples from open datasets like LibriSpeech (very short clips)
- Record your own test audio with clear licensing

### Adding Files

1. Ensure files are < 1MB each
2. Update this README with file descriptions
3. Add corresponding tests in `tests/integration/`
4. Mark integration tests with `@pytest.mark.integration`

### Git LFS (Optional)

For larger test files, consider using Git LFS:

```bash
git lfs track "tests/assets/audio/*"
```
