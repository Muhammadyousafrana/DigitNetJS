/**
 * AI Handwriting Classifier - Main Application
 * Enhanced with better security, performance, and error handling
 */

import { MnistData } from '/js/data.js';

// ============================================================================
// Application State
// ============================================================================

const APP_STATE = {
  model: null,
  data: null,
  canvas: null,
  ctx: null,
  rawImage: null,
  isDrawing: false,
  hasDrawn: false,
  isTraining: false,
  isModelReady: false,
  currentEpoch: 0,
  totalEpochs: 10,
  penSize: 24,
  visorOpen: false,
};

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  canvas: {
    width: 280,
    height: 280,
    lineWidth: 24,
    lineCap: 'round',
    strokeStyle: 'white',
    fillStyle: 'black',
  },
  model: {
    batchSize: 512,
    trainDataSize: 5500,
    testDataSize: 1000,
    epochs: 10,
    validationSplit: 0.15,
  },
  ui: {
    toastDuration: 3000,
    animationDuration: 300,
  },
};

// ============================================================================
// Model Architecture
// ============================================================================

function createModel() {
  try {
    const model = tf.sequential({
      name: 'digit-classifier',
    });

    // First Convolutional Block
    model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 3,
      filters: 32,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'conv1',
    }));

    model.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2],
      name: 'pool1',
    }));

    // Second Convolutional Block
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'conv2',
    }));

    model.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2],
      name: 'pool2',
    }));

    // Third Convolutional Block
    model.add(tf.layers.conv2d({
      filters: 128,
      kernelSize: 3,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'conv3',
    }));

    // Flatten and Dense Layers
    model.add(tf.layers.flatten({ name: 'flatten' }));

    model.add(tf.layers.dropout({
      rate: 0.3,
      name: 'dropout1',
    }));

    model.add(tf.layers.dense({
      units: 256,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'dense1',
    }));

    model.add(tf.layers.dropout({
      rate: 0.4,
      name: 'dropout2',
    }));

    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'dense2',
    }));

    model.add(tf.layers.dropout({
      rate: 0.4,
      name: 'dropout3',
    }));

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      name: 'dense3',
    }));

    model.add(tf.layers.dense({
      units: 10,
      activation: 'softmax',
      kernelInitializer: 'glorotNormal',
      name: 'output',
    }));

    const optimizer = tf.train.adam(0.001);

    model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    model.summary();
    return model;

  } catch (error) {
    console.error('❌ Error creating model:', error);
    showToast('Error creating model', error.message, 'error');
    throw error;
  }
}


// ============================================================================
// Training Functions
// ============================================================================

async function trainModel(model, data) {
  try {
    APP_STATE.isTraining = true;
    updateModelStatus('Training...');

    const { batchSize, trainDataSize, testDataSize, epochs } = CONFIG.model;

    // Prepare training data
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(trainDataSize);
      return [
        d.xs.reshape([trainDataSize, 28, 28, 1]),
        d.labels
      ];
    });

    // Prepare validation data
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(testDataSize);
      return [
        d.xs.reshape([testDataSize, 28, 28, 1]),
        d.labels
      ];
    });

    // Custom callback for training progress
    const customCallback = {
      onEpochEnd:  async (epoch, logs) => {
        APP_STATE.currentEpoch = epoch + 1;
        const progress = ((epoch + 1) / epochs) * 100;

        // Update UI with all metrics
        updateTrainingProgress(progress, epoch + 1, logs);

        // Allow UI to update
        await tf.nextFrame();
      },
      onBatchEnd: async (batch, logs) => {
        // Periodic UI updates during batch processing
        if (batch % 10 === 0) {
          await tf.nextFrame();
        }
      },
    };

    // TensorFlow Vis callbacks
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training',
      tab: 'Training',
      styles: { height: '640px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    // Combine callbacks
    const combinedCallbacks = {
      onEpochEnd: async (epoch, logs) => {
        // Call tfvis callback first
        if (fitCallbacks && fitCallbacks.onEpochEnd) {
          await fitCallbacks.onEpochEnd(epoch, logs);
        }
        // Then call a custom callback
        if (customCallback.onEpochEnd) {
          await customCallback.onEpochEnd(epoch, logs);
        }
      },
      onBatchEnd: async (batch, logs) => {
        if (fitCallbacks && fitCallbacks.onBatchEnd) {
          await fitCallbacks.onBatchEnd(batch, logs);
        }
        if (customCallback.onBatchEnd) {
          await customCallback.onBatchEnd(batch, logs);
        }
      },
    };

    // Show the visor and toggle button
    tfvis.visor().open();
    APP_STATE.visorOpen = true;
    updateVisorButtonText();

    const toggleBtn = document.getElementById('toggleVisorBtn');
    if (toggleBtn) {
      toggleBtn.classList.remove('hidden');
    }

    // Train the model
    const history = await model.fit(trainXs, trainYs, {
      batchSize,
      validationData: [testXs, testYs],
      epochs,
      shuffle: true,
      callbacks: combinedCallbacks,
    });

    // Clean up tensors
    trainXs.dispose();
    trainYs.dispose();
    testXs.dispose();
    testYs.dispose();

    APP_STATE.isTraining = false;
    APP_STATE.isModelReady = true;

    updateModelStatus('Ready');
    showToast('Training Complete!', 'Model is ready to classify digits', 'success');

    // Show drawing section
    showDrawingSection();

    return history;

  } catch (error) {
    console.error('❌ Training error:', error);
    APP_STATE.isTraining = false;
    updateModelStatus('Error');
    showToast('Training Failed', error.message, 'error');
    throw error;
  }
}

// ============================================================================
// Canvas Drawing Functions
// ============================================================================

function initCanvas() {
  APP_STATE.canvas = document.getElementById('canvas');
  APP_STATE.rawImage = document.getElementById('canvasimg');
  APP_STATE.ctx = APP_STATE.canvas. getContext('2d', {
    willReadFrequently: true,
  });

  // Set canvas properties
  const { ctx } = APP_STATE;
  const { fillStyle } = CONFIG.canvas;

  ctx.fillStyle = fillStyle;
  ctx.fillRect(0, 0, CONFIG.canvas.width, CONFIG.canvas.height);

  // Event listeners
  APP_STATE.canvas.addEventListener('mousedown', handleMouseDown);
  APP_STATE.canvas.addEventListener('mousemove', handleMouseMove);
  APP_STATE.canvas.addEventListener('mouseup', handleMouseUp);
  APP_STATE.canvas.addEventListener('mouseleave', handleMouseUp);

  // Touch support
  APP_STATE.canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
  APP_STATE.canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
  APP_STATE.canvas.addEventListener('touchend', handleTouchEnd);

  // Button event listeners
  document.getElementById('classifyBtn').addEventListener('click', classifyDrawing);
  document.getElementById('clearBtn').addEventListener('click', clearCanvas);

  // Pen size control
  const penSizeSlider = document. getElementById('penSize');
  const penSizeValue = document.getElementById('penSizeValue');
  const penPreview = document.getElementById('penPreview');

  penSizeSlider.addEventListener('input', (e) => {
    APP_STATE.penSize = parseInt(e. target.value);
    penSizeValue.textContent = APP_STATE.penSize;
    penPreview.style.width = `${APP_STATE.penSize}px`;
    penPreview.style.height = `${APP_STATE.penSize}px`;
  });

  // Initialize pen preview
  penPreview.style. width = `${APP_STATE.penSize}px`;
  penPreview.style.height = `${APP_STATE.penSize}px`;

  console.log('✅ Canvas initialized');
}

function handleMouseDown(e) {
  APP_STATE.isDrawing = true;
  const rect = APP_STATE.canvas. getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect. top;

  APP_STATE.ctx.beginPath();
  APP_STATE.ctx.moveTo(x, y);

  hideCanvasOverlay();
}

function handleMouseMove(e) {
  if (APP_STATE.isDrawing) {
    draw(e);
  }
}

function handleMouseUp() {
  if (APP_STATE.isDrawing) {
    APP_STATE.isDrawing = false;
    APP_STATE.rawImage.src = APP_STATE.canvas.toDataURL('image/png');
    document.getElementById('classifyBtn').disabled = false;
  }
}

function handleTouchStart(e) {
  e.preventDefault();
  APP_STATE.isDrawing = true;
  const touch = e.touches[0];
  const rect = APP_STATE. canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;

  APP_STATE.ctx.beginPath();
  APP_STATE.ctx.moveTo(x, y);

  hideCanvasOverlay();
}

function handleTouchMove(e) {
  e.preventDefault();
  if (APP_STATE.isDrawing) {
    const touch = e.touches[0];
    const rect = APP_STATE.canvas.getBoundingClientRect();
    const x = touch. clientX - rect.left;
    const y = touch.clientY - rect.top;

    drawAtPosition(x, y);
  }
}

function handleTouchEnd(e) {
  e.preventDefault();
  if (APP_STATE.isDrawing) {
    APP_STATE.isDrawing = false;
    APP_STATE.rawImage.src = APP_STATE.canvas.toDataURL('image/png');
    document.getElementById('classifyBtn').disabled = false;
  }
}

function draw(e) {
  if (!APP_STATE.isDrawing) return;

  const rect = APP_STATE. canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  drawAtPosition(x, y);
}

function drawAtPosition(x, y) {
  const { ctx } = APP_STATE;
  const { lineCap, strokeStyle } = CONFIG.canvas;

  ctx.lineWidth = APP_STATE.penSize;
  ctx.lineCap = lineCap;
  ctx.strokeStyle = strokeStyle;
  ctx.lineJoin = 'round';

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);

  APP_STATE.hasDrawn = true;
}

function clearCanvas() {
  const { ctx } = APP_STATE;
  const { fillStyle } = CONFIG.canvas;

  ctx.fillStyle = fillStyle;
  ctx.fillRect(0, 0, CONFIG.canvas.width, CONFIG.canvas.height);

  APP_STATE.hasDrawn = false;

  // Disable classify button
  document.getElementById('classifyBtn').disabled = true;

  // Hide prediction results
  document.getElementById('predictionResult').classList.remove('hidden');
  document.getElementById('predictionsList').classList.add('hidden');

  showCanvasOverlay();
}

// ============================================================================
// Prediction Functions
// ============================================================================

async function classifyDrawing() {
  if (!APP_STATE.isModelReady || !APP_STATE.hasDrawn) {
    showToast('Not Ready', 'Please draw a digit first', 'error');
    return;
  }

  try {
    // Prepare the image
    const tensor = tf.tidy(() => {
      // Convert image to tensor
      const raw = tf.browser.fromPixels(APP_STATE.rawImage, 1);

      // Resize to 28x28
      const resized = tf.image.resizeBilinear(raw, [28, 28]);

      // Normalize pixel values
      const normalized = resized.div(255.0);

      // Add batch dimension
      return normalized.expandDims(0);
    });

    // Make prediction
    const prediction = APP_STATE.model.predict(tensor);
    const probabilities = await prediction.data();

    // Get predicted class
    const predictedClass = await tf.argMax(prediction, 1).data();

    // Clean up tensors
    tensor.dispose();
    prediction.dispose();

    // Display results
    displayPredictions(predictedClass[0], probabilities);

    console.log('✅ Classification complete:', predictedClass[0]);

  } catch (error) {
    console.error('❌ Classification error:', error);
    showToast('Classification Failed', error.message, 'error');
  }
}

function displayPredictions(predictedDigit, probabilities) {
  // Hide placeholder
  document.getElementById('predictionResult').classList.add('hidden');

  // Show predictions list
  const predictionsList = document.getElementById('predictionsList');
  predictionsList.classList.remove('hidden');

  // Display predicted digit
  document.getElementById('predictedDigit').textContent = predictedDigit;

  // Create confidence bars
  const confidenceBars = document.getElementById('confidenceBars');
  confidenceBars.innerHTML = '';

  // Sort predictions by confidence
  const predictions = Array.from(probabilities)
    .map((prob, index) => ({ digit: index, confidence: prob }))
    .sort((a, b) => b.confidence - a.confidence);

  // Display all predictions
  predictions.forEach((pred, index) => {
    const item = document.createElement('div');
    item.className = 'confidence-item';
    item.style.animationDelay = `${index * 50}ms`;

    const percentage = (pred.confidence * 100).toFixed(1);
    const isTopPrediction = pred.digit === predictedDigit;

    item.innerHTML = `
      <span class="confidence-label">${pred.digit}</span>
      <div class="confidence-bar-container">
        <div class="confidence-bar-fill ${isTopPrediction ? 'top-prediction' : ''}"
             style="width: ${percentage}%">
          <span class="confidence-value">${percentage}%</span>
        </div>
      </div>
    `;

    confidenceBars.appendChild(item);
  });
}

// ============================================================================
// UI Update Functions
// ============================================================================

function updateModelStatus(status) {
  const statusElement = document.getElementById('modelStatus');
  if (statusElement) {
    statusElement. textContent = status;
  }
}

function updateTrainingProgress(percentage, epoch, logs) {
  // Update progress bar
  const progressFill = document.getElementById('progressFill');
  const progressPercentage = document.getElementById('progressPercentage');

  if (progressFill) {
    progressFill.style.width = `${percentage}%`;
  }

  if (progressPercentage) {
    progressPercentage.textContent = `${Math.round(percentage)}%`;
  }

  // Update epoch
  const currentEpoch = document.getElementById('currentEpoch');
  if (currentEpoch) {
    currentEpoch.textContent = `${epoch}/${CONFIG.model.epochs}`;
  }

  // Update accuracy - check both 'accuracy' and 'acc' keys
  const currentAccuracy = document.getElementById('currentAccuracy');
  if (currentAccuracy) {
    const accuracy = logs.accuracy || logs.acc || 0;
    currentAccuracy.textContent = `${(accuracy * 100).toFixed(2)}%`;
  }

  // Update validation accuracy
  const currentValAccuracy = document.getElementById('currentValAccuracy');
  if (currentValAccuracy) {
    const valAccuracy = logs.val_accuracy || logs.val_acc || 0;
    currentValAccuracy.textContent = `${(valAccuracy * 100).toFixed(2)}%`;
  }

  // Update loss
  const currentLoss = document.getElementById('currentLoss');
  if (currentLoss && logs.loss) {
    currentLoss.textContent = logs.loss. toFixed(4);
  }
}

function showWelcomeSection() {
  const welcomeSection = document.getElementById('welcomeSection');
  if (welcomeSection) {
    welcomeSection. classList.remove('hidden');
  }
}

function hideWelcomeSection() {
  const welcomeSection = document. getElementById('welcomeSection');
  if (welcomeSection) {
    welcomeSection.classList.add('hidden');
  }
}

function showTrainingSection() {
  const trainingSection = document.getElementById('trainingSection');
  if (trainingSection) {
    trainingSection.classList.remove('hidden');
  }
}

function showDrawingSection() {
  // Hide training section
  const trainingSection = document.getElementById('trainingSection');
  if (trainingSection) {
    trainingSection.classList.add('hidden');
  }

  // Show drawing section
  const drawingSection = document.getElementById('drawingSection');
  if (drawingSection) {
    drawingSection. classList.remove('hidden');
  }

  // Show architecture section
  const architectureSection = document.getElementById('architectureSection');
  if (architectureSection) {
    architectureSection.classList.remove('hidden');
  }
}

function hideCanvasOverlay() {
  const overlay = document.getElementById('canvasOverlay');
  if (overlay) {
    overlay.classList.add('hidden');
  }
}

function showCanvasOverlay() {
  const overlay = document.getElementById('canvasOverlay');
  if (overlay) {
    overlay.classList. remove('hidden');
  }
}

function updateVisorButtonText() {
  const toggleBtn = document.getElementById('toggleVisorBtn');
  if (toggleBtn) {
    const textSpan = toggleBtn.querySelector('span');
    if (textSpan) {
      textSpan.textContent = APP_STATE.visorOpen ? 'Hide Graphs' : 'Show Graphs';
    }
  }
}

// ============================================================================
// Toast Notifications
// ============================================================================

function showToast(title, message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document. createElement('div');
  toast.className = `toast ${type}`;

  const icons = {
    success: '✅',
    error: '❌',
    info: 'ℹ️',
    warning: '⚠️',
  };

  toast. innerHTML = `
    <span class="toast-icon">${icons[type] || icons.info}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
    </div>
  `;

  container.appendChild(toast);

  // Auto-remove after duration
  setTimeout(() => {
    toast.style.animation = 'slideOutRight 0.3s ease-out';
    setTimeout(() => {
      if (container.contains(toast)) {
        container. removeChild(toast);
      }
    }, CONFIG.ui.animationDuration);
  }, CONFIG.ui.toastDuration);
}

// ============================================================================
// Main Application Flow
// ============================================================================

async function initializeApp() {
  try {
    console.log('🚀 Initializing AI Handwriting Classifier...');
    updateModelStatus('Ready to start');

    // Load MNIST data in background
    updateModelStatus('Loading data...');
    APP_STATE.data = new MnistData();
    await APP_STATE.data.load();
    console.log('✅ Data loaded');

    updateModelStatus('Data loaded - Ready to train');
    showToast('Ready! ', 'Click "Start Training" to begin', 'success');

    // Setup start training button
    const startBtn = document.getElementById('startTrainingBtn');
    if (startBtn) {
      startBtn.addEventListener('click', startTraining);
    }

    // Setup toggle visor button
    const toggleVisorBtn = document.getElementById('toggleVisorBtn');
    if (toggleVisorBtn) {
      toggleVisorBtn.addEventListener('click', () => {
        tfvis.visor().toggle();
        APP_STATE.visorOpen = tfvis.visor().isOpen();
        updateVisorButtonText();
      });
    }

  } catch (error) {
    console.error('❌ Initialization error:', error);
    updateModelStatus('Error');
    showToast('Initialization Error', 'Failed to load data', 'error');
  }
}

async function startTraining() {
  try {
    // Disable start button
    const startBtn = document.getElementById('startTrainingBtn');
    if (startBtn) {
      startBtn.disabled = true;
      startBtn.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
        </svg>
        Starting...
      `;
    }

    // Hide welcome, show training
    hideWelcomeSection();
    showTrainingSection();

    updateModelStatus('Creating model...');

    // Create model
    APP_STATE.model = createModel();

    // Show model summary
    tfvis.show.modelSummary({
      name: 'Model Architecture',
      tab: 'Model'
    }, APP_STATE.model);

    updateModelStatus('Training model...');

    // Train model
    await trainModel(APP_STATE.model, APP_STATE.data);

    // Initialize canvas
    initCanvas();

    console.log('✅ Application ready');

  } catch (error) {
    console.error('❌ Training error:', error);
    updateModelStatus('Error');
    showToast('Training Error', 'Failed to train the model', 'error');

    // Re-enable start button
    const startBtn = document.getElementById('startTrainingBtn');
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        Start Training
      `;
    }
  }
}

// ============================================================================
// Memory Management
// ============================================================================

// Clean up resources on page unload
window.addEventListener('beforeunload', () => {
  if (APP_STATE.model) {
    APP_STATE.model.dispose();
  }
  tf.disposeVariables();
  console.log('🧹 Resources cleaned up');
});

// ============================================================================
// Initialize Application
// ============================================================================

document.addEventListener('DOMContentLoaded', initializeApp);

// Export for testing
export { createModel, classifyDrawing, clearCanvas };
