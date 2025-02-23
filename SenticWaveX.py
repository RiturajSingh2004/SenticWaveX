import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForSequenceClassification
import numpy as np
import sounddevice as sd
from threading import Thread, Event
import queue
import time
import sys
import signal
from datetime import datetime
import argparse
import os
import onnxruntime as ort
from pathlib import Path
import torch.nn as nn
from collections import Counter
import warnings

# Suppress unnecessary warnings and TensorFlow logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class AudioDeviceError(Exception):
    #Custom exception for audio device errors with error codes and suggestions.
    ERROR_CODES = {
        100: "Device not found",
        101: "Failed to open audio stream",
        102: "Unsupported audio format",
        103: "Permission denied to access audio device",
        104: "Unknown audio device error"
    }

    def __init__(self, message=None, error_code=104, suggestion=None):
        self.error_code = error_code
        self.message = message or self.ERROR_CODES.get(error_code, "Unknown error")
        self.suggestion = suggestion or "Please check your audio device settings and try again."
        full_message = f"[Error Code {self.error_code}] {self.message}\nSuggestion: {self.suggestion}"
        super().__init__(full_message)

    def log_error(self):
        ColoredOutput.print_error(f"AudioDeviceError: {self.message}")
        if self.suggestion:
            ColoredOutput.print_status(f"Suggestion: {self.suggestion}")

class ColoredOutput:
    #Utility class for colored terminal output with standardized message formatting.
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'

    @staticmethod
    def print_status(msg):
        print(f"{ColoredOutput.BLUE}[STATUS]{ColoredOutput.ENDC} {msg}")

    @staticmethod
    def print_success(msg):
        print(f"{ColoredOutput.GREEN}[SUCCESS]{ColoredOutput.ENDC} {msg}")

    @staticmethod
    def print_warning(msg):
        print(f"{ColoredOutput.WARNING}[WARNING]{ColoredOutput.ENDC} {msg}")

    @staticmethod
    def print_error(msg):
        print(f"{ColoredOutput.ERROR}[ERROR]{ColoredOutput.ENDC} {msg}")

class CustomWavLMModel(WavLMForSequenceClassification):
    
    #Custom WavLM model with Xavier weight initialization for improved training stability
    #and better convergence in emotion classification tasks.
   
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()

    def init_weights(self):
        #Initialize model weights using Xavier initialization for better performance.
        if hasattr(self, 'classifier'):
            nn.init.xavier_uniform_(self.classifier.weight)
            if hasattr(self.classifier, 'bias') and self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)
        
        if hasattr(self, 'projector'):
            nn.init.xavier_uniform_(self.projector.weight)
            if hasattr(self.projector, 'bias') and self.projector.bias is not None:
                nn.init.zeros_(self.projector.bias)

class OptimizedAudioEmotionAnalyzer:
    
    #Real-time audio emotion analyzer using WavLM model with optimizations for Snapdragon devices.
    #Supports continuous audio processing with confidence boosting and temporal consistency.
    
    def __init__(self, verbose=True, confidence_threshold=0.6):
        self.verbose = verbose
        self.stop_event = Event()
        self.processing_error_count = 0
        self.MAX_ERROR_COUNT = 5
        self.last_success_time = time.time()
        self.TIMEOUT_THRESHOLD = 30
        self.audio_queue = queue.Queue(maxsize=50)
        self.is_recording = False
        self.sample_rate = 16000  # Required sample rate for WavLM
        self.channels = 1  # Mono audio input
        self.block_size = 8000  # 0.5 seconds of audio at 16kHz
        self.confidence_threshold = confidence_threshold
        self.emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'surprised', 'fearful', 'disgusted']
        self.emotion_history = []
        self.HISTORY_SIZE = 5
        
        try:
            self._initialize_audio_device()
            self._initialize_optimized_model()
        except Exception as e:
            ColoredOutput.print_error(f"Initialization error: {str(e)}")
            raise

    def _initialize_audio_device(self):
        #Initialize audio device with proper error handling and device selection.
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            ColoredOutput.print_status("Available audio input devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"  {i}: {device['name']}")
            self.device = default_input['index']
            ColoredOutput.print_status(f"Using default input device: {default_input['name']}")

            # Test audio stream to ensure device works
            with sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size
            ) as stream:
                ColoredOutput.print_success("Audio device test successful")
        except sd.PortAudioError as e:
            raise AudioDeviceError(f"Failed to open audio stream: {str(e)}")
        except Exception as e:
            raise AudioDeviceError(f"Audio device initialization failed: {str(e)}")

    def _initialize_optimized_model(self):
        
        #Initialize and optimize WavLM model with ONNX runtime for Snapdragon devices
        #or standard PyTorch for other platforms.
        
        if self.verbose:
            ColoredOutput.print_status("Loading and optimizing WavLM model...")
            
        try:
            cache_dir = Path("./model_cache")
            cache_dir.mkdir(exist_ok=True)
            
            # Initialize feature extractor with consistent settings
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus",
                cache_dir=cache_dir,
                padding_side="right",
                return_attention_mask=True
            )
            
            # Configure model with proper number of emotion classes
            from transformers import WavLMConfig
            config = WavLMConfig.from_pretrained(
                "microsoft/wavlm-base-plus",
                num_labels=len(self.emotion_labels),
                cache_dir=cache_dir
            )
            
            model = CustomWavLMModel(config)
            model.wavlm = WavLMForSequenceClassification.from_pretrained(
                "microsoft/wavlm-base-plus",
                config=config,
                cache_dir=cache_dir
            ).wavlm
            
            model.config.use_cache = False
            
            # Optimize for Snapdragon devices using ONNX runtime
            if self._is_snapdragon():
                from torch.nn.utils import parametrizations
                parametrized_model = parametrizations.weight_norm(model)
                model = parametrized_model.eval()
                
                onnx_path = cache_dir / "wavlm_optimized.onnx"
                if not onnx_path.exists():
                    self._export_onnx_model(model, onnx_path)
                
                self._initialize_onnx_session(onnx_path)
                ColoredOutput.print_success("Model optimized for Qualcomm hardware")
            else:
                self.model = model.eval()
            
            self._initialize_confidence_boosting()
            
        except Exception as e:
            raise Exception(f"Model optimization failed: {str(e)}")

    def _export_onnx_model(self, model, onnx_path):
        #Export PyTorch model to ONNX format with proper input/output configuration.
        dummy_input = torch.randn(1, 8000)
        dummy_attention_mask = torch.ones(1, 8000, dtype=torch.long)
        torch.onnx.export(
            model,
            (dummy_input, dummy_attention_mask),
            onnx_path,
            input_names=['input', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'}
            }
        )

    def _initialize_onnx_session(self, onnx_path):
        #Initialize ONNX runtime session with optimizations for Snapdragon.
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = str(onnx_path)
        
        providers = ['QNNExecutionProvider'] if 'QNNExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
            sess_options=sess_options
        )

    def _is_snapdragon(self):
        #Detect Snapdragon processor for optimized inference.
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                return 'snapdragon' in cpuinfo or 'qualcomm' in cpuinfo
        except Exception:
            return False

    def _initialize_confidence_boosting(self):
        
        #Initialize confidence boosting parameters for improved emotion prediction accuracy.
        #Uses weighted factors and temporal consistency through transition matrix.
        
        self.confidence_weights = {
            'signal_strength': 0.3,  # Weight for audio signal strength
            'temporal_consistency': 0.4,  # Weight for temporal emotion consistency
            'acoustic_quality': 0.3  # Weight for audio quality metrics
        }
       
        # Probability transition matrix for emotion states
        self.transition_matrix = np.array([
            [0.6, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025],  # angry
            [0.1, 0.6, 0.05, 0.15, 0.05, 0.025, 0.025],  # happy
            [0.1, 0.05, 0.6, 0.15, 0.05, 0.025, 0.025],  # sad
            [0.1, 0.15, 0.15, 0.4, 0.1, 0.05, 0.05],    # neutral
            [0.05, 0.05, 0.05, 0.1, 0.6, 0.075, 0.075],  # surprised
            [0.025, 0.025, 0.025, 0.05, 0.075, 0.7, 0.1],# fearful
            [0.025, 0.025, 0.025, 0.05, 0.075, 0.1, 0.7] # disgusted
        ])

    def _boost_confidence(self, predictions, audio_data):
        #Boost confidence using multiple factors
        # Convert predictions to numpy if it's a tensor
        emotion_probs = predictions.detach().numpy() if isinstance(predictions, torch.Tensor) else predictions
        
        # Ensure emotion_probs is 1D
        emotion_probs = emotion_probs.squeeze()
        
        # Signal strength factor
        signal_strength = np.abs(audio_data).mean()
        signal_factor = min(signal_strength / 0.1, 1.0)
        
        # Temporal consistency
        temporal_factor = np.ones_like(emotion_probs)  # Initialize with ones
        if self.emotion_history:
            last_emotion = self.emotion_history[-1]
            last_emotion_idx = self.emotion_labels.index(last_emotion)
            temporal_factor = self.transition_matrix[last_emotion_idx]
        
        # Acoustic quality
        acoustic_factor = self._estimate_acoustic_quality(audio_data)
        
        # Combine factors
        boosted_probs = (
            emotion_probs * self.confidence_weights['signal_strength'] * signal_factor +
            temporal_factor * self.confidence_weights['temporal_consistency'] +
            acoustic_factor * self.confidence_weights['acoustic_quality']
        )
        
        # Ensure no negative values
        boosted_probs = np.maximum(boosted_probs, 0)
        
        # Normalize probabilities
        sum_probs = boosted_probs.sum()
        if sum_probs > 0:
            return boosted_probs / sum_probs
        else:
            return np.ones_like(boosted_probs) / len(boosted_probs)  # Uniform distribution as fallback
    def _estimate_acoustic_quality(self, audio_data):
        #Estimate audio quality using basic metrics
        signal = np.abs(audio_data)
        noise = np.abs(audio_data - np.mean(audio_data))
        snr = np.mean(signal) / (np.mean(noise) + 1e-6)
        return min(snr / 10.0, 1.0)

    def predict_emotion(self, audio_data):
        #Predict emotion from audio data with correct tensor dimensions
        try:
            # Convert to numpy if it's not already
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.numpy()
            
            # Ensure audio_data is the right shape for WavLM
            audio_data = audio_data.squeeze()  # Remove any extra dimensions
            
            # If it's a mono channel, reshape to [sequence_length]
            if len(audio_data.shape) == 2:  # If it's [frames, channels]
                audio_data = audio_data[:, 0]  # Take first channel if stereo
            
            # Process audio
            inputs = self.processor(
                audio_data,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
            
            if hasattr(self, 'ort_session'):
                ort_inputs = {
                    'input': inputs['input_values'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy()
                }
                ort_outputs = self.ort_session.run(None, ort_inputs)
                predictions = torch.nn.functional.softmax(torch.tensor(ort_outputs[0]), dim=-1)
            else:
                with torch.no_grad():
                    outputs = self.model(
                        input_values=inputs['input_values'],
                        attention_mask=inputs['attention_mask']
                    )
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predictions = predictions.squeeze()
            if len(predictions.shape) == 0:
                predictions = predictions.unsqueeze(0)
            
            if len(predictions) != len(self.emotion_labels):
                raise ValueError(f"Prediction dimension mismatch: got {len(predictions)}, expected {len(self.emotion_labels)}")
            
            boosted_predictions = self._boost_confidence(predictions, audio_data)
            emotion_idx = int(np.argmax(boosted_predictions))
            confidence = float(boosted_predictions[emotion_idx])
            predicted_emotion = self.emotion_labels[emotion_idx]
            
            return predicted_emotion, confidence
            
        except Exception as e:
            self.processing_error_count += 1
            if self.verbose:
                ColoredOutput.print_error(f"Prediction error: {str(e)}")
                import traceback
                traceback.print_exc()
            time.sleep(0.1)
            return "error", 0.0

    def audio_callback(self, indata, frames, time_info, status):
        #Handle incoming audio data with correct dimensions
        if status:
            ColoredOutput.print_warning(f"Audio callback status: {status}")
            return
            
        try:
            if np.any(indata):
                self.last_success_time = time.time()
                
                # Normalize the audio data
                normalized_data = indata.copy()
                if np.max(np.abs(normalized_data)) > 0:
                    normalized_data = normalized_data / np.max(np.abs(normalized_data))
                
                # Take first channel if stereo
                if normalized_data.shape[1] > 1:
                    normalized_data = normalized_data[:, 0]
                
                try:
                    self.audio_queue.put(normalized_data, block=False)
                except queue.Full:
                    pass
                    
        except Exception as e:
            if self.verbose:
                ColoredOutput.print_error(f"Audio callback error: {str(e)}")
            time.sleep(0.1)

    def start_realtime_analysis(self):
        #Start real-time analysis with 10-second limit
        self.is_recording = True
        self.stop_event.clear()
        self.last_success_time = time.time()
        start_time = time.time()
        last_print_time = 0
        MIN_PRINT_INTERVAL = 0.1
        RECORDING_DURATION = 10
        buffer = []  # Buffer to accumulate audio chunks
        
        try:
            ColoredOutput.print_status("Starting audio stream...")
            ColoredOutput.print_status(f"Recording will stop after {RECORDING_DURATION} seconds...")
            
            with sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=self.block_size
            ) as stream:
                ColoredOutput.print_success("Audio stream started")
                
                while not self.stop_event.is_set() and self.is_recording:
                    try:
                        if time.time() - start_time >= RECORDING_DURATION:
                            ColoredOutput.print_status("\nRecording duration completed (10 seconds)")
                            self.stop_realtime_analysis()
                            break
                            
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        buffer.append(audio_chunk)
                        
                        # Process when we have enough data (e.g., 1 second worth)
                        if len(buffer) * self.block_size >= self.sample_rate:
                            # Concatenate buffer
                            audio_data = np.concatenate(buffer)
                            buffer = []  # Clear buffer
                            
                            if np.abs(audio_data).mean() > 0.01:
                                emotion, confidence = self.predict_emotion(audio_data)
                                
                                current_time = time.time()
                                if current_time - last_print_time >= MIN_PRINT_INTERVAL:
                                    if emotion != "error":
                                        remaining_time = RECORDING_DURATION - (current_time - start_time)
                                        timestamp = datetime.now().strftime("%H:%M:%S")
                                        print(f"\r{ColoredOutput.BLUE}[{timestamp}]{ColoredOutput.ENDC} "
                                            f"Emotion: {emotion.ljust(10)} "
                                            f"Confidence: {confidence:.2%} "
                                            f"Time remaining: {remaining_time:.1f}s", end="")
                                        sys.stdout.flush()
                                        last_print_time = current_time
                                
                    except queue.Empty:
                        continue
                    except Exception as e:
                        if self.verbose:
                            ColoredOutput.print_error(f"\nAnalysis error: {str(e)}")
                            traceback.print_exc()
                        time.sleep(0.1)
                        continue                        
        except Exception as e:
            ColoredOutput.print_error(f"Stream error: {str(e)}")
        finally:
            self.stop_realtime_analysis()
            ColoredOutput.print_status("\nRecording stopped")
    
    def stop_realtime_analysis(self):
        #Stop the real-time analysis
        self.is_recording = False
        self.stop_event.set()
        
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        ColoredOutput.print_status("Analysis stopped")

def main():
    parser = argparse.ArgumentParser(description='Real-time Audio Emotion Analysis')
    parser.add_argument('--duration', type=int, default=0, help='Duration to run the analysis (in seconds), 0 for unlimited')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--device', type=int, help='Audio input device index (optional)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, 
                        help='Confidence threshold for emotion detection (0.0-1.0)')
    parser.add_argument('--history-size', type=int, default=5,
                        help='Size of emotion history buffer for smoothing')
    args = parser.parse_args()

    try:
        analyzer = OptimizedAudioEmotionAnalyzer(
            verbose=args.verbose,
            confidence_threshold=args.confidence_threshold
        )
        if args.device is not None:
            analyzer.device = args.device
        
        analyzer.HISTORY_SIZE = args.history_size

        def signal_handler(signum, frame):
            ColoredOutput.print_status("\nStopping gracefully...")
            analyzer.stop_realtime_analysis()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        analysis_thread = Thread(target=analyzer.start_realtime_analysis)
        analysis_thread.start()

        if args.duration > 0:
            time.sleep(args.duration)
            analyzer.stop_realtime_analysis()
        
        analysis_thread.join()

    except AudioDeviceError as e:
        e.log_error()
        ColoredOutput.print_status("Try running with --device to specify a different audio input device")
        sys.exit(1)
    except Exception as e:
        ColoredOutput.print_error(f"Fatal error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()