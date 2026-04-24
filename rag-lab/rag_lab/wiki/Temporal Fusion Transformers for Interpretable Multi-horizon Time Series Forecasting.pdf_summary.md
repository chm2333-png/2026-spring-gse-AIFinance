# Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.pdf

**Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**
==============================================

### Authors
* Bryan Lima
* Sercan Ö. Arık
* Nicolas Loeff
* Tomas Pfüster

### Research Question
The research question addressed in this paper is how to develop a deep learning model that can perform multi-horizon time series forecasting while providing interpretable insights into the temporal dynamics of the data. The authors aim to design a model that can handle a complex mix of inputs, including static covariates, known future inputs, and other exogenous time series, without any prior information on how they interact with the target.

### Methodology
The authors propose a novel attention-based architecture called the Temporal Fusion Transformer (TFT), which combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics. The TFT model uses:

* Recurrent layers for local processing
* Interpretable self-attention layers for long-term dependencies
* Specialized components to select relevant features
* Gating layers to suppress unnecessary components

### Key Findings
The authors demonstrate significant performance improvements over existing benchmarks on a variety of real-world datasets. Specifically, they report:

* Improved forecasting accuracy compared to state-of-the-art models
* Ability to identify globally-important variables for the prediction problem
* Ability to detect persistent temporal patterns
* Ability to identify significant events

### Data Sources Used
The authors use a variety of real-world datasets, including:

* Retail data
* Healthcare data
* Economic data

### Policy Implications
The TFT model has several policy implications, including:

* Improved decision-making in retail, healthcare, and economics through more accurate multi-horizon forecasting
* Ability to identify key factors driving forecasted outcomes, enabling more targeted interventions
* Potential to improve resource allocation and optimization in various industries

### Connections to Other Research
The TFT model builds on existing research in:

* [[Deep Learning]] for time series forecasting
* [[Attention Mechanisms]] for interpretable modeling
* [[Time Series Analysis]] for understanding temporal dynamics
* [[Explainable AI]] for developing transparent and trustworthy models

The authors also draw connections to other research areas, including:

* [[Recurrent Neural Networks]] (RNNs) for sequential data modeling
* [[Transformer Models]] for natural language processing and other applications
* [[Interpretable Machine Learning]] for developing models that provide insights into their decision-making processes.