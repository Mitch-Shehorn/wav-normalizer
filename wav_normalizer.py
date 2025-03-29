#!/usr/bin/env python3
"""
WAV File Volume Normalizer

This script batch normalizes WAV files to a specified peak level (default: -0.1dB)
to ensure consistent volume across all samples.

Usage:
    python wav_normalizer.py [--dir DIR] [--target TARGET] [--prefix PREFIX] [--method METHOD] [--dry-run]

Arguments:
    --dir DIR           Directory containing WAV files (default: current directory)
    --target TARGET     Target peak level in dB (default: -0.1)
    --prefix PREFIX     Prefix for output files (default: "norm_")
    --method METHOD     Normalization method: peak, rms (default: peak)
    --dry-run           Analyze without modifying files
"""

import os
import glob
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm

def db_to_linear(db):
    """Convert dB value to linear gain factor"""
    return 10 ** (db / 20)

def linear_to_db(linear):
    """Convert linear gain factor to dB"""
    if linear <= 0:
        return -np.inf
    return 20 * np.log10(linear)

def analyze_audio(audio_data):
    """
    Analyze audio data to get peak and RMS levels
    
    Parameters:
        audio_data (numpy.ndarray): Audio data
        
    Returns:
        dict: Analysis results with peak and RMS values
    """
    # Handle mono/stereo
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        # For stereo/multi-channel, find peak across all channels
        peak_abs = np.max(np.abs(audio_data))
        channel_count = audio_data.shape[1]
        
        # Calculate RMS per channel and use the maximum
        rms_values = []
        for ch in range(channel_count):
            rms_values.append(np.sqrt(np.mean(audio_data[:, ch] ** 2)))
        rms_linear = max(rms_values)
    else:
        # For mono
        flat_audio = audio_data.flatten()
        peak_abs = np.max(np.abs(flat_audio))
        rms_linear = np.sqrt(np.mean(flat_audio ** 2))
        channel_count = 1
    
    peak_db = linear_to_db(peak_abs)
    rms_db = linear_to_db(rms_linear)
    
    # Calculate crest factor (dynamic range)
    crest_factor_db = peak_db - rms_db if rms_db > -np.inf else np.inf
    
    return {
        'channel_count': channel_count,
        'peak_linear': peak_abs,
        'peak_db': peak_db,
        'rms_linear': rms_linear,
        'rms_db': rms_db,
        'crest_factor_db': crest_factor_db
    }

def normalize_audio(audio_data, target_db=-0.1, method='peak'):
    """
    Normalize audio data to target level
    
    Parameters:
        audio_data (numpy.ndarray): Audio data to normalize
        target_db (float): Target peak level in dB
        method (str): Normalization method ('peak' or 'rms')
        
    Returns:
        tuple: (normalized_audio, gain_db, analysis_before, analysis_after)
    """
    # Analyze current levels
    analysis_before = analyze_audio(audio_data)
    
    # Convert target dB to linear
    target_linear = db_to_linear(target_db)
    
    # Determine reference level based on method
    if method == 'rms':
        current_level = analysis_before['rms_linear']
    else:  # Default to peak
        current_level = analysis_before['peak_linear']
    
    # Calculate gain factor
    if current_level > 0:
        gain_linear = target_linear / current_level
        gain_db = linear_to_db(gain_linear)
    else:
        # If audio is complete silence, no gain needed
        return audio_data.copy(), 0, analysis_before, analysis_before
    
    # Apply gain
    normalized_audio = audio_data * gain_linear
    
    # Analyze after normalization
    analysis_after = analyze_audio(normalized_audio)
    
    return normalized_audio, gain_db, analysis_before, analysis_after

def process_wav_file(input_file, output_file, target_db=-0.1, method='peak', dry_run=False):
    """
    Process a single WAV file by normalizing its volume
    
    Parameters:
        input_file (str): Path to input WAV file
        output_file (str): Path to output normalized WAV file
        target_db (float): Target peak level in dB
        method (str): Normalization method ('peak' or 'rms')
        dry_run (bool): If True, only analyze without saving
        
    Returns:
        dict: Processing results
    """
    try:
        # Read audio file with SoundFile
        audio_data, sample_rate = sf.read(input_file)
        
        # Get file info for later (to preserve bit depth)
        info = sf.info(input_file)
        
        # Normalize audio
        normalized_audio, gain_applied_db, before_analysis, after_analysis = normalize_audio(
            audio_data, target_db, method
        )
        
        result = {
            'filename': input_file,
            'before': before_analysis,
            'after': after_analysis,
            'gain_applied_db': gain_applied_db
        }
        
        # Save normalized audio if not in dry-run mode
        if not dry_run:
            # Use same subtype as original to preserve bit depth
            sf.write(output_file, normalized_audio, sample_rate, subtype=info.subtype)
            result['output_file'] = output_file
        
        return result
        
    except Exception as e:
        return {
            'filename': input_file,
            'error': str(e)
        }

def format_db(db_value):
    """Format dB value for display"""
    if db_value == -np.inf:
        return "-∞ dB"
    return f"{db_value:.2f} dB"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Normalize WAV files to consistent volume level.')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing WAV files')
    parser.add_argument('--target', type=float, default=-0.1, help='Target peak level in dB')
    parser.add_argument('--prefix', type=str, default='norm_', help='Prefix for output files')
    parser.add_argument('--method', type=str, default='peak', 
                        choices=['peak', 'rms'],
                        help='Normalization method')
    parser.add_argument('--dry-run', action='store_true', help='Analyze without modifying files')
    args = parser.parse_args()
    
    # Find all WAV files in the specified directory
    search_pattern = os.path.join(args.dir, '*.wav')
    wav_files = glob.glob(search_pattern)
    wav_files.extend(glob.glob(search_pattern.replace('.wav', '.WAV')))
    
    if not wav_files:
        print(f"No WAV files found in directory: {args.dir}")
        return
    
    print(f"Found {len(wav_files)} WAV files in {args.dir}")
    print(f"Target level: {args.target} dB")
    print(f"Method: {args.method.upper()} normalization")
    
    if args.dry_run:
        print("DRY RUN MODE: Files will not be modified")
    
    # Process each file
    results = []
    for input_file in tqdm(wav_files, desc="Processing files"):
        file_dir, file_name = os.path.split(input_file)
        output_file = os.path.join(file_dir, f"{args.prefix}{file_name}")
        
        print(f"\nProcessing: {os.path.basename(input_file)}")
        
        result = process_wav_file(
            input_file, 
            output_file, 
            target_db=args.target,
            method=args.method,
            dry_run=args.dry_run
        )
        
        # Display results
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            before_peak = format_db(result['before']['peak_db'])
            after_peak = format_db(result['after']['peak_db'])
            before_rms = format_db(result['before']['rms_db'])
            after_rms = format_db(result['after']['rms_db'])
            gain = format_db(result['gain_applied_db'])
            
            print(f"  Channels: {result['before']['channel_count']}")
            print(f"  Before: Peak: {before_peak}, RMS: {before_rms}")
            print(f"  After:  Peak: {after_peak}, RMS: {after_rms}")
            print(f"  Gain applied: {gain}")
            
            if abs(result['gain_applied_db']) < 0.1:
                print("  ℹ️ File already at target level (no significant change)")
        
        results.append(result)
    
    # Summarize results
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    significant_changes = sum(1 for r in results 
                            if 'gain_applied_db' in r and abs(r['gain_applied_db']) >= 3)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total WAV files: {total}")
    print(f"Files with errors: {errors}")
    
    if not args.dry_run:
        print(f"Files normalized: {total - errors}")
    
    print(f"Files with significant level changes (≥3dB): {significant_changes}")
    
    if significant_changes > 0:
        print("\nFiles with significant level changes:")
        for result in results:
            if 'gain_applied_db' in result and abs(result['gain_applied_db']) >= 3:
                filename = os.path.basename(result['filename'])
                gain = format_db(result['gain_applied_db'])
                print(f"  - {filename}: {gain}")

if __name__ == "__main__":
    main()
