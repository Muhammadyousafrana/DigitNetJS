/**
 * MNIST Data Loader
 * Enhanced with better error handling and validation
 * @license Apache-2.0
 */

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const TRAIN_TEST_RATIO = 5 / 6;

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

/**
 * A class that fetches the MNIST dataset and provides shuffled batches
 * with enhanced error handling and validation
 */
export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
    this.datasetImages = null;
    this.datasetLabels = null;
    this. trainIndices = null;
    this. testIndices = null;
    this.trainImages = null;
    this.testImages = null;
    this. trainLabels = null;
    this.testLabels = null;
  }

  /**
   * Load the MNIST dataset with validation and error handling
   */
  async load() {
    try {
      console.log('📥 Loading MNIST dataset...');

      // Load images and labels in parallel
      const [imageData, labelData] = await Promise.all([
        this.loadImagesWithRetry(),
        this.loadLabelsWithRetry()
      ]);

      this.datasetImages = imageData;
      this.datasetLabels = labelData;

      // Validate data
      this.validateData();

      // Create shuffled indices
      this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
      this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

      // Split into train and test sets
      this. trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
      this.testImages = this.datasetImages. slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
      this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
      this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);

      console.log('✅ MNIST dataset loaded successfully');
      console.log(`   Training samples: ${NUM_TRAIN_ELEMENTS}`);
      console.log(`   Test samples: ${NUM_TEST_ELEMENTS}`);

    } catch (error) {
      console.error('❌ Error loading MNIST dataset:', error);
      throw new Error(`Failed to load MNIST dataset: ${error.message}`);
    }
  }

  /**
   * Load MNIST images with retry logic
   */
  async loadImagesWithRetry(retries = MAX_RETRIES) {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        return await this.loadImages();
      } catch (error) {
        console.warn(`⚠️ Image load attempt ${attempt} failed:`, error.message);

        if (attempt === retries) {
          throw error;
        }

        // Wait before retrying
        await this.delay(RETRY_DELAY * attempt);
      }
    }
  }

  /**
   * Load MNIST labels with retry logic
   */
  async loadLabelsWithRetry(retries = MAX_RETRIES) {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        return await this.loadLabels();
      } catch (error) {
        console.warn(`⚠️ Labels load attempt ${attempt} failed:`, error.message);

        if (attempt === retries) {
          throw error;
        }

        // Wait before retrying
        await this.delay(RETRY_DELAY * attempt);
      }
    }
  }

  /**
   * Load MNIST images sprite
   */
  async loadImages() {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const canvas = document. createElement('canvas');
      const ctx = canvas.getContext('2d');

      img.crossOrigin = 'anonymous';

      img.onload = () => {
        try {
          img.width = img.naturalWidth;
          img.height = img.naturalHeight;

          const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
          const chunkSize = 5000;

          canvas.width = img.width;
          canvas.height = chunkSize;

          for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
            const datasetBytesView = new Float32Array(
              datasetBytesBuffer,
              i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize
            );

            ctx.drawImage(
              img,
              0,
              i * chunkSize,
              img.width,
              chunkSize,
              0,
              0,
              img.width,
              chunkSize
            );

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            for (let j = 0; j < imageData.data. length / 4; j++) {
              // All channels hold an equal value since the image is grayscale
              // Just read the red channel
              datasetBytesView[j] = imageData.data[j * 4] / 255;
            }
          }

          const datasetImages = new Float32Array(datasetBytesBuffer);
          resolve(datasetImages);

        } catch (error) {
          reject(new Error(`Failed to process images: ${error.message}`));
        }
      };

      img.onerror = () => {
        reject(new Error('Failed to load MNIST images sprite'));
      };

      // Add timeout
      const timeout = setTimeout(() => {
        reject(new Error('Image load timeout'));
      }, 30000);

      img.onload = ((originalOnload) => {
        return function() {
          clearTimeout(timeout);
          originalOnload.apply(this, arguments);
        };
      })(img.onload);

      img.src = MNIST_IMAGES_SPRITE_PATH;
    });
  }

  /**
   * Load MNIST labels
   */
  async loadLabels() {
    try {
      const response = await fetch(MNIST_LABELS_PATH);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const labelsData = await response.arrayBuffer();
      return new Uint8Array(labelsData);

    } catch (error) {
      throw new Error(`Failed to load MNIST labels: ${error.message}`);
    }
  }

  /**
   * Validate loaded data
   */
  validateData() {
    if (!this.datasetImages || this.datasetImages.length === 0) {
      throw new Error('Dataset images are empty or invalid');
    }

    if (!this.datasetLabels || this.datasetLabels.length === 0) {
      throw new Error('Dataset labels are empty or invalid');
    }

    const expectedImageSize = NUM_DATASET_ELEMENTS * IMAGE_SIZE;
    if (this.datasetImages.length !== expectedImageSize) {
      throw new Error(
        `Invalid image data size. Expected ${expectedImageSize}, got ${this.datasetImages. length}`
      );
    }

    const expectedLabelSize = NUM_DATASET_ELEMENTS * NUM_CLASSES;
    if (this.datasetLabels.length !== expectedLabelSize) {
      throw new Error(
        `Invalid label data size. Expected ${expectedLabelSize}, got ${this.datasetLabels.length}`
      );
    }

    console.log('✅ Data validation passed');
  }

  /**
   * Get next training batch
   */
  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  /**
   * Get next test batch
   */
  nextTestBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.testImages, this.testLabels],
      () => {
        this.shuffledTestIndex = (this. shuffledTestIndex + 1) % this.testIndices.length;
        return this.testIndices[this.shuffledTestIndex];
      }
    );
  }

  /**
   * Get next batch of data
   */
  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image = data[0]. slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray. set(image, i * IMAGE_SIZE);

      const label = data[1]. slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
  }

  /**
   * Utility function to delay execution
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
