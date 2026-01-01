"""
Audio Player Module

Simple audio playback using pygame for playing hotspot sound effects.
"""

import os
import threading

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Audio playback will be disabled.")


class AudioPlayer:
    """Plays audio files for hotspot interactions."""
    
    def __init__(self, sounds_dir: str = "assets/sounds"):
        """
        Initialize the audio player.
        
        Args:
            sounds_dir: Directory containing sound files.
        """
        self.sounds_dir = sounds_dir
        self.enabled = PYGAME_AVAILABLE
        self.sounds_cache = {}
        self.currently_playing = None
        self._lock = threading.Lock()
        
        if self.enabled:
            try:
                pygame.mixer.init()
                self._preload_sounds()
            except Exception as e:
                print(f"Warning: Could not initialize pygame.mixer: {e}")
                print("Audio playback will be disabled.")
                self.enabled = False
    
    def _preload_sounds(self):
        """Preload all sound files from the sounds directory."""
        if not os.path.isdir(self.sounds_dir):
            print(f"Warning: Sounds directory not found: {self.sounds_dir}")
            return
        
        for filename in os.listdir(self.sounds_dir):
            if filename.endswith('.mp3'):
                name = os.path.splitext(filename)[0]
                path = os.path.join(self.sounds_dir, filename)
                try:
                    self.sounds_cache[name] = pygame.mixer.Sound(path)
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")
        
        print(f"Loaded {len(self.sounds_cache)} sounds from {self.sounds_dir}")
    
    def play(self, name: str):
        """
        Play a sound by name.
        
        Args:
            name: Name of the sound (without extension).
        """
        if not self.enabled:
            return
        
        with self._lock:
            # Don't interrupt if already playing the same sound
            if self.currently_playing == name and pygame.mixer.get_busy():
                return
            
            if name in self.sounds_cache:
                pygame.mixer.stop()  # Stop any currently playing sound
                self.sounds_cache[name].play()
                self.currently_playing = name
            else:
                print(f"Sound not found: {name}")
    
    def stop(self):
        """Stop any currently playing sound."""
        if self.enabled:
            pygame.mixer.stop()
            self.currently_playing = None
    
    def cleanup(self):
        """Release audio resources."""
        if self.enabled:
            pygame.mixer.quit()
