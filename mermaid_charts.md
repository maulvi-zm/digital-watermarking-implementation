# Mermaid Charts for Program Flow

## Chart 1: Overall Experiment Pipeline

```mermaid
flowchart TD
    Start([Start Experiment]) --> LoadData[Load MNIST Dataset]
    LoadData --> TrainBaseline[Train Baseline Model]
    TrainBaseline --> GenWatermark[Generate Watermark & Key]
    GenWatermark --> TrainWatermarked[Train Watermarked Model]
    TrainWatermarked --> VerifyWM[Verify Watermark]
    VerifyWM --> TestFineTune[Fine-tuning Attack]
    VerifyWM --> TestPruning[Pruning Attack]
    TestFineTune --> EvalRobust[Evaluate Robustness]
    TestPruning --> EvalRobust
    EvalRobust --> SaveResults[Save Results]
    SaveResults --> End([End])
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style TrainBaseline fill:#fff4e1
    style TrainWatermarked fill:#fff4e1
    style VerifyWM fill:#e8f5e9
    style TestFineTune fill:#ffebee
    style TestPruning fill:#ffebee
```

## Chart 2: Watermark Embedding Process

```mermaid
flowchart LR
    A[Model Parameters] --> B[Select Target Layer]
    C[Generate Binary Watermark] --> D[Create Secret Key]
    B --> E[Extract Parameters]
    D --> E
    E --> F[Compute Watermark Loss]
    G[Task Loss CrossEntropy] --> H[Combine Losses]
    F --> H
    H --> I[Backpropagation]
    I --> J[Update Parameters]
    J --> K[Watermarked Model]
    
    style C fill:#e8f5e9
    style F fill:#fff4e1
    style K fill:#e1f5ff
```

## Chart 3: Watermark Extraction and Verification

```mermaid
flowchart TD
    Start([Load Watermarked Model]) --> LoadKey[Load Secret Key]
    LoadKey --> ExtractParams[Extract Target Layer Parameters]
    ExtractParams --> DecodeWM[Decode Watermark]
    LoadKey --> DecodeWM
    DecodeWM --> Compare[Compare with Original]
    Compare --> CalcMetrics[Calculate Metrics]
    CalcMetrics --> BitAcc[Bit Accuracy]
    CalcMetrics --> ExactMatch[Exact Match]
    CalcMetrics --> HammingDist[Hamming Distance]
    BitAcc --> Result([Verification Result])
    ExactMatch --> Result
    HammingDist --> Result
    
    style Start fill:#e1f5ff
    style Result fill:#e1f5ff
    style DecodeWM fill:#fff4e1
    style Compare fill:#e8f5e9
```

## Chart 4: Robustness Testing Flow

```mermaid
flowchart TD
    Start([Watermarked Model]) --> Branch{Attack Type}
    Branch -->|Fine-tuning| FineTune[Retrain on Subset]
    Branch -->|Pruning| Prune[Remove Low-Magnitude Params]
    FineTune --> EvalAcc1[Evaluate Accuracy]
    Prune --> EvalAcc2[Evaluate Accuracy]
    EvalAcc1 --> ExtractWM1[Extract Watermark]
    EvalAcc2 --> ExtractWM2[Extract Watermark]
    ExtractWM1 --> Compare1[Compare with Original]
    ExtractWM2 --> Compare2[Compare with Original]
    Compare1 --> Survived1{Survived?}
    Compare2 --> Survived2{Survived?}
    Survived1 -->|Yes| Result1[Robust]
    Survived1 -->|No| Result2[Not Robust]
    Survived2 -->|Yes| Result1
    Survived2 -->|No| Result2
    Result1 --> End([End])
    Result2 --> End
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style FineTune fill:#ffebee
    style Prune fill:#ffebee
    style Result1 fill:#e8f5e9
    style Result2 fill:#ffcdd2
```

## Chart 5: Training Process with Watermark Embedding

```mermaid
flowchart TD
    Init[Initialize Model] --> LoadData[Load Training Data]
    LoadData --> Forward[Forward Pass]
    Forward --> TaskLoss[Compute Task Loss]
    Forward --> ExtractParams[Extract Target Parameters]
    ExtractParams --> WatermarkLoss[Compute Watermark Loss]
    TaskLoss --> CombineLoss[Total Loss = Task + λ × Watermark]
    WatermarkLoss --> CombineLoss
    CombineLoss --> Backward[Backward Pass]
    Backward --> Update[Update Parameters]
    Update --> CheckEpoch{Epoch < Max?}
    CheckEpoch -->|Yes| LoadData
    CheckEpoch -->|No| SaveModel[Save Watermarked Model]
    SaveModel --> End([Training Complete])
    
    style Init fill:#e1f5ff
    style End fill:#e1f5ff
    style WatermarkLoss fill:#fff4e1
    style CombineLoss fill:#e8f5e9
```

## Chart 6: Complete System Architecture

```mermaid
graph TB
    subgraph DataLayer[Data Layer]
        MNIST[MNIST Dataset]
    end
    
    subgraph ModelLayer[Model Layer]
        CNN[Simple CNN]
    end
    
    subgraph WatermarkLayer[Watermark Layer]
        Generator[Watermark Generator]
        Embedder[Watermark Embedder]
        Extractor[Watermark Extractor]
    end
    
    subgraph AttackLayer[Attack Layer]
        FineTune[Fine-tuning]
        Pruning[Parameter Pruning]
    end
    
    subgraph EvalLayer[Evaluation Layer]
        Fidelity[Model Fidelity]
        Reliability[Watermark Reliability]
        Robustness[Robustness Metrics]
    end
    
    MNIST --> CNN
    Generator --> Embedder
    Embedder --> CNN
    CNN --> Extractor
    CNN --> FineTune
    CNN --> Pruning
    FineTune --> Robustness
    Pruning --> Robustness
    CNN --> Fidelity
    Extractor --> Reliability
    
    style DataLayer fill:#e3f2fd
    style ModelLayer fill:#fff3e0
    style WatermarkLayer fill:#e8f5e9
    style AttackLayer fill:#ffebee
    style EvalLayer fill:#f3e5f5
```

