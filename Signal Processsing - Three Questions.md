## Signal Processing - Three Questions

# Question 1

Components include:

1. **Sensors/Transducers**
   - **Function:** Converts physical phenomena (e.g., temperature, pressure,
     light) into electrical signals.
   - **Limitations:**
     - Sensitivity and accuracy can degrade over time due to environmental
       conditions.
     - Calibration drift can require frequent recalibration.
     - Non-linearity can lead to errors in measurement if outside the
       specified range.

2. **Data Acquisition System**
   - **Function:** Converts analog signals into digital form for processing.
   - **Limitations:**
     - Limited sampling rate might not capture fast moving signals
       accurately.
     - Limited resolution can lead to quantization errors.

3. **Data Processing and Analysis**
   - **Function:** Software tools and algorithms used to analyze, interpret,
     and visualize data. This can include data cleaning, data reduction,
     feature extraction, etc.
   - **Limitations:**
     - Errors in the algorithms can lead to incorrect data interpretation.
     - Processing may require significant computational resources.

# Question 2

**Sampling Rate:** To accurately reconstruct a signal without aliasing, the
sampling rate must be at least twice the highest frequency present in the
signal (Nyquist rate).

**Length of the Signal:** 
    - **Frequency Resolution:** A longer signal length allows for better
      frequency resolution in spectral analysis, as the frequency resolution
      is inversely proportional to the signal length when using methods like
      the Fourier Transform.
    - **Statistical Significance:** Longer signals provide more data points,
      leading to robust statistical analysis and reducing the effect of
      noise and random fluctuations.

# Question 3

1. **Signal Analyzer:**
   - **Exploratory Data Analysis:** Visualizes and explores time-series to
      understand its characteristics, such as trends, noise levels, and
      frequency components.
   - **Preprocessing:** Performs operations like filtering, smoothing, and
     resampling to cleanse and prepare the signal data for ML modeling.
   - **Spectral Analysis:** Analyzes the frequency domain to uncover
     important features or patterns.
   - **Feature Generation:** Identifies features derived from the time or
     frequency domain that can be used as inputs for ML modeling.

2. **Diagnostic Feature Designer:**
   - **Automated Feature Extraction:** Generates and evaluates various
     features from the signal data, such as statistical metrics, shape-based
     features, and frequency-domain features.
   - **Feature Selection and Ranking:** Automatically ranks and selects the
     most relevant features based on their diagnostic performance.
