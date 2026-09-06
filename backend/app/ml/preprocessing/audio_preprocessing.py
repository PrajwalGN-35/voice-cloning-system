"""
Audio Preprocessing Pipeline for Voice Spoofing Detection

This module provides reusable functions for audio preprocessing,
designed to work with AASIST and other voice spoofing detection models.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Union, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Handles audio loading, preprocessing, and feature extraction for voice spoofing detection.
    
    Attributes:
        sample_rate (int): Target sample rate (default: 16000 Hz)
        n_mels (int): Number of mel-frequency bins (default: 128)
        n_fft (int): FFT size (default: 512)
        hop_length (int): Number of samples between successive frames (default: 160)
    """
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, 
                 n_fft: int = 512, hop_length: int = 160, 
                 duration: float = 5.0):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate (int): Target sample rate in Hz
            n_mels (int): Number of mel-frequency bins
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            duration (float): Audio duration in seconds (for padding/truncation)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.max_samples = int(sample_rate * duration)
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path (str): Path to audio file (WAV, FLAC, MP3, etc.)
            
        Returns:
            np.ndarray: Audio waveform (mono, normalized to [-1, 1])
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If audio file cannot be loaded
        """
        try:
            # Load audio and resample to target sample rate
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            logger.info(f"✓ Loaded audio: {audio_path} (duration: {len(audio)/self.sample_rate:.2f}s)")
            return audio
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio (np.ndarray): Audio waveform
            
        Returns:
            np.ndarray: Normalized audio
        """
        # Remove any DC offset
        audio = audio - np.mean(audio)
        
        # Normalize by peak amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad or truncate audio to fixed duration.
        
        Args:
            audio (np.ndarray): Audio waveform
            
        Returns:
            np.ndarray: Audio padded/truncated to fixed length
        """
        if len(audio) < self.max_samples:
            # Pad with zeros
            pad_amount = self.max_samples - len(audio)
            audio = np.pad(audio, (0, pad_amount), mode='constant', constant_values=0)
            logger.info(f"✓ Padded audio to {len(audio)} samples")
        else:
            # Truncate
            audio = audio[:self.max_samples]
            logger.info(f"✓ Truncated audio to {len(audio)} samples")
        
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio waveform.
        
        Args:
            audio (np.ndarray): Audio waveform
            
        Returns:
            np.ndarray: Mel-spectrogram (n_mels, time_steps)
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (in dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        logger.info(f"✓ Extracted mel-spectrogram: {mel_spec_db.shape}")
        return mel_spec_db
    
    def preprocess(self, audio_path: str, verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline: load → normalize → pad/truncate → extract mel-spec.
        
        Args:
            audio_path (str): Path to audio file
            verbose (bool): Print progress messages
            
        Returns:
            Tuple containing:
                - np.ndarray: Processed mel-spectrogram (128, ~157)
                - dict: Metadata with original duration, sample rate, shape
        """
        # Load and normalize
        audio = self.load_audio(audio_path)
        audio = self.normalize_audio(audio)
        
        # Pad/truncate
        audio = self.pad_or_truncate(audio)
        
        # Extract mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Collect metadata
        metadata = {
            'original_duration': len(audio) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'mel_spec_shape': mel_spec.shape,
            'n_mels': self.n_mels,
            'audio_path': audio_path
        }
        
        if verbose:
            logger.info(f"✓ Preprocessing complete!\n  Shape: {mel_spec.shape}\n  Duration: {metadata['original_duration']:.2f}s")
        
        return mel_spec, metadata


def preprocess_audio_file(audio_path: str, sample_rate: int = 16000, 
                         duration: float = 5.0) -> Tuple[np.ndarray, dict]:
    """
    Standalone function to preprocess a single audio file.
    
    Args:
        audio_path (str): Path to audio file
        sample_rate (int): Target sample rate
        duration (float): Target duration in seconds
        
    Returns:
        Tuple of (mel_spectrogram, metadata_dict)
    """
    preprocessor = AudioPreprocessor(sample_rate=sample_rate, duration=duration)
    return preprocessor.preprocess(audio_path)
