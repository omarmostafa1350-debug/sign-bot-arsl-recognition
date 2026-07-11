# Sign-Bot Architecture Diagrams: Mermaid Source Code

This file contains the raw Mermaid definitions for the 5 key system diagrams. You can copy and paste these blocks directly into the [Mermaid Live Editor](https://mermaid.live) or any markdown compiler supporting Mermaid rendering.

---

## 1. High-Level System Architecture (Artifact 1)

```text
graph TD
    %% Training Phase
    subgraph Training Phase [Google Colab T4 GPU]
        A1[KArSL-502 Video Dataset] -->|MP Hands model_complexity=0| B1(Landmark Extraction: extract_hands.py)
        B1 -->|Raw 126-dim Features| C1(Dataset Builder: build_dataset.py)
        C1 -->|Shared Norm| D1(Preprocessing Core: preprocessing.py)
        C1 -->|Padded/Truncated 20-frame seqs| E1[Consolidated Dataset ZIP]
        E1 -->|Load Dataset & Train| F1(BiLSTM Training: train_production_model.py)
        F1 -->|Export Checkpoint| G1[signbot_production_best.h5]
        G1 -->|Direct Keras Model Conversion| H1(TFLite Converter: convert_tflite_simple.py)
        H1 -->|Dynamic Range Quantization| I1[signbot_production.tflite]
    end

    %% Deployment & Runtime Phase
    subgraph Deployment Phase [Raspberry Pi 4 / Local Mac CPU]
        I1 -->|Copy TFLite Model| J2[TFLite Interpreter]
        K2[Webcam Capture Feed] -->|Frames 640x480| L2(Resolution Resize 320x240)
        L2 -->|MediaPipe Hands| M2(Landmark Detector)
        M2 -->|Extract Raw Landmarks| N2[Flat Array 126-dim]
        N2 -->|Shared Norm| O2(Preprocessing Core: preprocessing.py)
        O2 -->|Sliding Window Buffer: maxlen=20| P2(collections.deque)
        P2 -->|Every 3 Frames| J2
        J2 -->|Softmax Probability Vector| Q2(Post-processing Confidence Gates)
        Q2 -->|Arabic Shaping & RTL Layout| R2(Text Rendering: reshaper & bidi)
        R2 -->|Draw Text Glyph Overlay| S2[PyQt5 GUI / Frame Display]
    end
```

---

## 2. Software Module Architecture Class Diagram (Artifact 2)

```text
classDiagram
    class ConfigTraining {
        +SSD_ROOT : String
        +NPY_DIR : String
        +DATASET_DIR : String
        +NUM_CLASSES : int = 100
        +MAX_SEQUENCE_LENGTH : int = 20
        +FRAME_WIDTH : int = 320
        +FRAME_HEIGHT : int = 240
    }
    class PreprocessingCore {
        +normalize_hand_landmarks(hand_row, num_points)
        +normalize_frame(frame_hands)
        +normalize_sequence(seq_hands)
    }
    class ExtractHands {
        +find_videos()
        +extract_landmarks_from_video(video_path, hands_detector)
    }
    class BuildDataset {
        +pad_or_truncate(sequence, max_len)
        +collect_samples()
        +build_arrays(samples, split_name)
    }
    class TrainProductionModel {
        +load_signer(signer_id, split)
        +build_model()
    }
    class ConvertTfliteSimple {
        +load_model()
        +_strip_quantization_config(obj)
    }
    class TestWebcamLive {
        +draw_text(frame_bgr, text, position)
        +extract_and_normalize_hands(results)
    }

    ExtractHands --> ConfigTraining : imports
    BuildDataset --> ConfigTraining : imports
    BuildDataset --> PreprocessingCore : imports
    TrainProductionModel --> PreprocessingCore : imports
    ConvertTfliteSimple --> ConfigTraining : imports
    TestWebcamLive --> PreprocessingCore : imports
```

---

## 3. Runtime Flowchart (Artifact 3)

```text
flowchart TD
    A[Start Runtime Script] --> B[Load TFLite Model & Class Names]
    B --> C[Initialize OpenCV VideoCapture & MediaPipe Hands]
    C --> D[Initialize collections.deque maxlen=20]
    D --> E[Read Frame from Camera]
    E -->|Success?| F{Yes}
    E -->|Failure?| G[Log Error & Exit]
    F --> H[Resize frame to 320x240]
    H --> I[Convert Frame to RGB & Run MediaPipe Hands]
    I --> J{Hands Detected?}
    
    J -->|Yes| K[Extract 126-dim Landmarks]
    J -->|No| L[Fill 126-dim Vector with Zeros]
    
    K --> M{Was prev frame empty?}
    M -->|Yes| N[Clear Deque Buffer]
    M -->|No| O[Apply preprocessing.normalize_frame]
    N --> O
    L --> O
    
    O --> P[Append Normalized Frame to Deque]
    P --> Q{Is Deque Buffer Full = 20?}
    
    Q -->|No| R[Render Listening State on HUD]
    Q -->|Yes| S{Frame Count % 3 == 0?}
    
    S -->|No| R
    S -->|Yes| T[Invoke TFLite Interpreter]
    T --> U[Retrieve Softmax Prediction Class & Confidence]
    
    U --> V{Confidence >= 0.80?}
    V -->|Yes| W[Prediction Text = Arabic Word]
    V -->|No| X{Confidence >= 0.60?}
    
    X -->|Yes| Y[Prediction Text = ~Arabic Word]
    X -->|No| Z[Prediction Text = ...]
    
    W --> AA[Shape Arabic Letter Forms: arabic_reshaper]
    Y --> AA
    Z --> AA
    
    AA --> AB[Reorder for RTL Display: python-bidi]
    AB --> AC[PIL ImageDraw Text Glyph Overlay on Frame]
    AC --> AD[Display BGR Frame to Screen via OpenCV / PyQt]
    AD --> AE{Key 'q' Pressed?}
    AE -->|Yes| AF[Release Camera & Close Windows]
    AE -->|No| E
    R --> AA
```

---

## 4. Level 1 Data Flow Diagram (Artifact 8)

```text
graph TB
    subgraph Input & Capture
        D1[Camera Video Stream] -->|BGR Image Frame| P1(P1: Frame Resizing)
        P1 -->|320x240 Frame| P2(P2: Feature Extraction)
    end
    
    subgraph Feature Processing
        P2 -->|Raw Landmark Coordinates| P3(P3: Translation & Scale Normalization)
        P3 -->|Normalized Frame Vector| P4(P4: Sequence Buffering)
        P4 -->|Full 20x126 Sequence| P5(P5: TFLite Inference)
    end
    
    subgraph Classification & Formatting
        P5 -->|Class Index & Confidence| P6(P6: Post-Filtering & Confidence Gating)
        P6 -->|Arabic Word Label| P7(P7: Text Reshaping & BiDi Reordering)
        P7 -->|RTL Unicode String| P8(P8: HUD Graphics Overlay)
    end

    %% Data Stores
    TFLiteStore[(TFLite Weight Tensor Constants)] -->|Read weights| P5
    LabelStore[(class_names.npy)] -->|Lookup text| P6
    FontStore[(GeezaPro.ttc Glyph map)] -->|Map glyphs| P8
    
    P8 -->|Overlaid Display Frame| OutputDisplay[Screen Display]
```

---

## 5. Raspberry Pi 4 Deployment Architecture (Artifact 9)

```text
graph TD
    %% Hardware Components
    subgraph Hardware Layer [Raspberry Pi 4 Model B]
        Camera[Raspberry Pi Camera Module V2] -->|640x480 BGR Feed| CPU[Broadcom BCM2711 ARM CPU]
        CPU -->|SPI/HDMI| Screen[Humanoid Robot LCD Panel]
    end

    %% Software execution layers within the CPU
    subgraph OS & Runtime Layer [Debian Linux / ARMv8]
        subgraph MediaPipe Detector
            CPU -->|Frame resizing 320x240| MP[MediaPipe Hands C++ Backend]
        end

        subgraph Preprocessing
            MP -->|126-dim Landmark coords| Norm[preprocessing.py normalization]
        end

        subgraph Sequence Queue
            Norm -->|Append| Queue[collections.deque maxlen=20]
        end

        subgraph Model Inference
            Queue -->|Evaluate every 3rd Frame| Interp[TFLite Interpreter]
            FlexLib[libtensorflowlite_flex.so] -.->|Resolves SELECT_TF_OPS| Interp
        end

        subgraph Text Formatting & Render
            Interp -->|Softmax Prediction Index| ArabicSh[arabic_reshaper & python-bidi]
            ArabicSh -->|PIL ImageDraw| Overlay[GUI Image Overlay]
        end

        subgraph UI Thread
            Overlay -->|Render| PyQt[PyQt5 GUI Dashboard]
        end
    end

    PyQt -->|Video Output stream| Screen
```
