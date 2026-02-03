# DigitNetJS

AI Handwriting Classifier — a small browser-based digit classifier built with TensorFlow.js that lets users draw digits on a canvas and see model predictions in real time.

DigitNetJS includes:
- A MNIST data loader (MnistData) that fetches the dataset from Google Cloud and provides shuffled batches.
- A compact TensorFlow.js model implemented in-browser (dense layers + dropout + softmax).
- A responsive UI and drawing canvas for drawing, classifying, and visualizing predictions and confidence.
- Webpack development and production configurations for local development and builds.

---

## Table of contents

- <a>Demo</a>
- <a>Features</a>
- <a>Tech stack</a>
- <a>Project structure</a>
- <a>How it works</a>
  - <a>Data loader (MnistData)</a>
  - <a>Model architecture</a>
  - <a>Training &amp; Inference flow</a>
- <a>Getting started (local development)</a>
  - <a>Prerequisites</a>
  - <a>Install dependencies</a>
  - <a>Run dev server</a>
  - <a>Build for production</a>
  - <a>Serve production build</a>
- <a>Usage (UI)</a>
- <a>Testing &amp; exported functions</a>
- <a>Contributing</a>
- <a>Troubleshooting</a>
- <a>Roadmap</a>
- <a>License &amp; acknowledgements</a>
- <a>Contact</a>

---

## Demo

Open `index.html` in a browser or run the dev server (see below) and draw a digit on the canvas. Click "Classify Digit" to run the model on your drawing. The app displays the predicted digit and confidence bars for all classes.

---

## Features

- Client-side handwriting digit classification (MNIST-style).
- In-browser training support (UI for training, progress/metrics displayed).
- Visualization of prediction confidences and model status.
- Responsive UI and drawing canvas with touch &amp; mouse support.
- Webpack dev server configuration for fast iteration and production build config with asset copying.

---

## Tech stack

- JavaScript (ES Modules)
- HTML &amp; CSS (responsive UI)
- TensorFlow.js (via CDN)
- Webpack (dev &amp; prod configs)
- Optional: tfjs-vis for visualization (included via CDN in index.html)

CDNs included in `index.html`:
- https://cdn.jsdelivr.net/npm/@tensorflow/tfjs
- https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis

---

## Project structure (important files)

- index.html — main UI and script includes
- js/
  - app.js — main application logic, model creation, training, canvas handling, UI updates
  - data.js — MnistData class: dataset fetching and batch utilities
- css/style.css — styling for the UI and canvas
- webpack.common.js, webpack.config.dev.js, webpack.config.prod.js — build configs
- LICENSE.txt — project license (per repo)

---

## How it works

### Data loader (MnistData)
- The `MnistData` class in `js/data.js` fetches the MNIST image sprite and labels from:
  - images: `https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png`
  - labels: `https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8`
- It exposes methods to get shuffled train/test batches and includes retry logic and basic validation.
- Designed to supply flattened 28x28 images (784 features) and one-hot labels for 10 classes.

### Model architecture
- The model is created in `js/app.js` via the `createModel()` function.
- High-level architecture (summary):
  - Input: flattened 784 inputs (28x28)
  - Dense layers with ReLU activations and dropout layers for regularization
  - Final dense output layer with softmax for 10-class classification
  - Optimizer: Adam (learning rate ~0.001)
  - Loss: categorical crossentropy
  - Metrics: accuracy
- The model is compact to run efficiently in the browser and to allow training on small subsets of MNIST.

### Training &amp; Inference flow
- `trainModel(model, data)` handles training using batches from `MnistData`. Training progress (epochs, loss, accuracy) is surfaced to the UI.
- `classifyDrawing()` handles inference: it downscales/processes the canvas content into the same 28x28 grayscale input format expected by the model, then runs `model.predict()` and displays results.
- UI helpers handle drawing, touch/mouse events, clearing canvas and showing toast/notifications.

---

## Getting started (local development)

### Prerequisites
- Node.js (LTS) and npm or yarn
- Optional: an HTTP server to serve the `dist/` folder for production build

### Install dependencies
This repository uses Webpack for development and production builds. There might not be a package.json in the repo; if you don't have one, create it and add the dependencies below. Example dependencies you may want to install:

npm:
```bash
npm init -y
npm install --save-dev webpack webpack-cli webpack-dev-server webpack-merge html-webpack-plugin copy-webpack-plugin
```

The project also uses TensorFlow.js via CDN in `index.html`, so TensorFlow.js does not need to be installed as a local dependency unless you prefer local installation.

Optionally add tfjs packages:
```bash
npm install @tensorflow/tfjs @tensorflow/tfjs-vis
```

### Recommended scripts (add to package.json)
Add these scripts to your `package.json` for convenience:
```json
"scripts": {
  "dev": "webpack serve --config webpack.config.dev.js --mode development",
  "build": "webpack --config webpack.config.prod.js --mode production",
  "start": "npm run build && npx http-server ./dist -p 8080"
}
```
(You can replace `http-server` with any static server of your choice.)

### Run dev server
Start the dev server for live reloading and faster iteration:
```bash
npm run dev
```
This uses the `webpack.config.dev.js` devServer settings to serve the project and open the browser.

### Build for production
Create a production build (bundles assets into `dist/`):
```bash
npm run build
```
`webpack.config.prod.js` copies static assets (img, css, js/vendor, favicon, robots, etc.) into `dist/` and generates the production HTML output.

### Serve production build
After `npm run build`, serve the `dist/` folder with a static server:
```bash
npx http-server ./dist -p 8080
# open http://localhost:8080
```

---

## Usage (UI)

- Draw a digit using mouse or touch on the canvas.
- Click "Classify Digit" to run the model and see:
  - Predicted digit (top prediction)
  - Confidence bars for all 10 classes
- Click "Clear Canvas" to erase and draw again.
- If training is enabled in UI, you can start training and watch training progress/metrics.

UI notes:
- Model status indicator shows if model is "Initializing", "Ready", "Training", etc.
- There may be a toggle to show/hide training visualizations (tfjs-vis) in the header.

---

## Testing &amp; exported functions

`js/app.js` exports a few functions intended for testing and re-use:

- `createModel()` — returns a compiled TensorFlow.js model matching the app architecture.
- `classifyDrawing()` — runs inference on the current canvas and returns predicted digit + probabilities.
- `clearCanvas()` — clears the canvas programmatically.

These exports allow unit tests (e.g., using Jest with `tfjs-node` or browser-based testing) to validate the model creation, classification output shape, and UI canvas behavior.

---

## Contributing

Contributions are welcome! Suggested workflow:
1. Fork the repo and create a feature branch.
2. Implement changes with clear commit messages.
3. Ensure linting and basic manual testing (draw a digit, classify).
4. Open a pull request with a description of changes and any screenshots or GIFs showing UI behavior.

Before submitting training/accuracy changes, include:
- A summary of the change (e.g., model architecture, optimizer).
- Expected impact on performance or inference speed.
- Repro steps to validate.

---

## Troubleshooting

- Canvas stays black / model returns NaNs:
  - Ensure canvas preprocessing scales pixel values to the same range used during training and that input shape matches the model (784-long vector for 28x28).
- MNIST resources fail to download:
  - The MnistData loader fetches from external Google Cloud URLs; ensure your environment has outbound internet access.
  - If you want offline usage, host the MNIST resources locally and update the paths in `js/data.js`.
- Dev server does not start:
  - Confirm `webpack`, `webpack-dev-server`, and related packages are installed and `package.json` scripts are present.

---

## Roadmap (ideas)

- Add pre-trained model download so users can skip training and run inference immediately.
- Export/import trained model (tfjs.save) and a small UI to manage saved models.
- Improve mobile drawing responsiveness and undo/redo strokes.
- Add automated tests for model outputs and UI flows.
- Expand dataset augmentation and experiment with convolutional models for better accuracy.

---

## License &amp; acknowledgements

This repository includes a LICENSE.txt derived from the HTML5 Boilerplate license text included in the repo; please review `LICENSE.txt` for full terms. If you include other third-party assets, comply with their licenses.

Acknowledgements:
- MNIST dataset served from Google Cloud (used by the MnistData loader).
- TensorFlow.js and tfjs-vis teams for the client-side ML libraries.

---

## Contact

Maintainer: Muhammadyousafrana

If you need help or want to discuss improvements, open an issue or a pull request with details and steps to reproduce.

---

Thank you for checking out DigitNetJS — a compact, educational, and interactive demo for in-browser handwriting recognition. If you'd like, I can also add a package.json and recommended devDependencies to make running the project easier.