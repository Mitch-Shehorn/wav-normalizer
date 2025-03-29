//wav-normalizer basic information
- Created for drum one-shots. Run this in a folder of .wav audio files. It creates new copies of the audio files. The new copies have their volume level normalized to 0db.

//Install required dependancies
- python
- pip install numpy soundfile tqdm

 //Run the program
- place wav_normalizer.py in the folder with the audio files that you want to be normalized
- execute "python wav_normalizer.py" in the command prompt (that has been directed to the same folder)

//Advanced options
- Normalize explicitly to 0dB (max level)
  - python wav_normalizer.py --target 0.0
- Normalize to -3dB for additional headroom
  - python wav_normalizer.py --target -3.0
- Analyze files without modifying them (dry run)
  - python wav_normalizer.py --dry-run
- Use RMS normalization for more consistent perceived loudness
  - python wav_normalizer.py --method rms
- Specify a custom prefix for output files
  - python wav_normalizer.py --prefix "0db_"
- Process files in a different directory
  - python wav_normalizer.py --dir /path/to/your/drum/samples

//Core Functionality
- Volume Analysis: For each WAV file, the program calculates:
  - Peak amplitude (the absolute maximum sample value)
  - RMS level (the average energy of the signal)
  - Channel count (preserves stereo or mono configuration)
- Normalization Process:
  - Calculates the required gain to bring each file to the target level
  - Applies precise gain scaling to maintain audio quality
  - Preserves the original bit depth of your samples
  - Creates a new file with normalized volume (original files remain untouched)
- Multi-channel Handling:
  - Correctly processes both mono and stereo files
  - Normalizes based on the loudest channel in stereo files
  - Maintains proper stereo imaging and channel balance

//Technical Implementation Details
- The default target level is -0.1dB (slightly below 0dB) to prevent potential clipping
- Uses peak normalization by default, which is ideal for drum samples where transients are important
- Optionally supports RMS normalization, which can create more consistent perceived loudness
- Preserves metadata and file format specifications from the original files
