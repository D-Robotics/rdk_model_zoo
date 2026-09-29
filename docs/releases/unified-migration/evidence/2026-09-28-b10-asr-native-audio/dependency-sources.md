# Host audio dependency sources

- https://github.com/libsndfile/libsndfile/releases/download/1.2.2/libsndfile-1.2.2.tar.xz
- https://github.com/libsndfile/libsamplerate/releases/download/0.2.2/libsamplerate-0.2.2.tar.xz

Observed archive SHA-256 and actual configure/build/install commands are in dependencies.json. Downloaded sources and installed libraries remain in the ignored .coordination/asr-audio-deps directory; no third-party binary is vendored into the sample. This build supports the tested WAV inputs and does not certify the board's SDK or library versions.
