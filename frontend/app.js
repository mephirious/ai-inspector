// Global state
let currentDocId = null;
let combinedResults = null; // { "doc.pdf": {...}, "image.jpg": {...} }
let modelView = null; // processed: { documents, pagesByDoc, pageData }
let selectedDoc = null;
let selectedPageIndex = 0;
let visibleCategories = { signature: true, stamp: true, qr: true };

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const toggleSignature = document.getElementById('toggleSignature');
const toggleStamp = document.getElementById('toggleStamp');
const toggleQR = document.getElementById('toggleQR');
const downloadJsonBtn = document.getElementById('downloadJsonBtn');
const downloadZipBtn = document.getElementById('downloadZipBtn');
const fileListEl = document.getElementById('fileList');
const docTitleEl = document.getElementById('docTitle');
const prevPageBtn = document.getElementById('prevPageBtn');
const nextPageBtn = document.getElementById('nextPageBtn');
const pageIndicatorEl = document.getElementById('pageIndicator');
const pageImageEl = document.getElementById('pageImage');
const overlayCanvasEl = document.getElementById('overlayCanvas');

// API base URL
const API_BASE = window.location.origin;

// Event listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
uploadArea.addEventListener('dragleave', handleDragLeave);
fileInput.addEventListener('change', handleFileSelect);
uploadBtn.addEventListener('click', handleUpload);
toggleSignature.addEventListener('change', () => { visibleCategories.signature = toggleSignature.checked; renderCurrentPage(); });
toggleStamp.addEventListener('change', () => { visibleCategories.stamp = toggleStamp.checked; renderCurrentPage(); });
toggleQR.addEventListener('change', () => { visibleCategories.qr = toggleQR.checked; renderCurrentPage(); });
downloadJsonBtn.addEventListener('click', handleDownloadJson);
downloadZipBtn.addEventListener('click', handleDownloadZip);
prevPageBtn.addEventListener('click', () => changePage(-1));
nextPageBtn.addEventListener('click', () => changePage(1));

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
}

function handleFileSelect() {
    if (fileInput.files.length > 0) {
        uploadBtn.disabled = false;
        uploadArea.classList.add('has-file');
        const names = Array.from(fileInput.files).map(f => f.name);
        uploadArea.querySelector('.upload-text').textContent = `Selected: ${names.length} files`;
    } else {
        uploadBtn.disabled = true;
        uploadArea.classList.remove('has-file');
    }
}

async function handleUpload() {
    if (!fileInput.files.length) return;

    // Show loading
    loadingIndicator.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorMessage.classList.add('hidden');
    uploadBtn.disabled = true;

    try {
        const data = await FileUploadHandler.uploadFiles(fileInput.files);
        currentDocId = data.document_id;
        combinedResults = data.results; // combined JSON object
        modelView = processModelOutput(combinedResults);
        // Select first document with pages, if any
        selectedDoc = modelView.documents.find(doc => (modelView.pagesByDoc[doc] || []).length > 0) || null;
        selectedPageIndex = 0;
        renderUI();
    } catch (error) {
        showError(error.message);
    } finally {
        loadingIndicator.classList.add('hidden');
        uploadBtn.disabled = false;
    }
}

function renderUI() {
  // Populate sidebar file list
  fileListEl.innerHTML = '';
  for (const docName of modelView.documents) {
    const pages = modelView.pagesByDoc[docName] || [];
    // Skip docs with 0 pages
    const item = document.createElement('div');
    item.className = 'file-item' + (docName === selectedDoc ? ' active' : '');
    item.textContent = `${docName}${pages.length ? ` (${pages.length})` : ''}`;
    item.addEventListener('click', () => {
      selectedDoc = docName;
      selectedPageIndex = 0;
      renderUI();
    });
    fileListEl.appendChild(item);
  }

  // Update header and controls
  if (!selectedDoc || (modelView.pagesByDoc[selectedDoc] || []).length === 0) {
    docTitleEl.textContent = 'Select a document';
    pageIndicatorEl.textContent = '-/-';
    resultsSection.classList.remove('hidden');
    return;
  }

  docTitleEl.textContent = selectedDoc;
  renderCurrentPage();
  resultsSection.classList.remove('hidden');
}

function renderCurrentPage() {
  if (!selectedDoc) return;
  const pages = modelView.pagesByDoc[selectedDoc] || [];
  if (pages.length === 0) return;
  // Clamp page index
  if (selectedPageIndex < 0) selectedPageIndex = 0;
  if (selectedPageIndex >= pages.length) selectedPageIndex = pages.length - 1;
  const pageId = pages[selectedPageIndex];
  const pageData = modelView.pageData[selectedDoc][pageId];

  // Update indicator
  pageIndicatorEl.textContent = `${selectedPageIndex + 1}/${pages.length} • ${pageId}`;

  // Set image src and draw overlay
  const imgSrc = `${API_BASE}/images/${currentDocId}/${pageId}?original=true`;
  PageRenderer.setImageAndCanvas(pageImageEl, overlayCanvasEl, imgSrc, pageData.page_size)
    .then(() => {
      PageRenderer.drawAnnotations(overlayCanvasEl, pageData.annotations, visibleCategories);
      renderDetectionsList(pageData.annotations);
    })
    .catch(() => {
      showError('Failed to load page image');
    });
}

function renderDetectionsList(annotations) {
  const listEl = document.getElementById('detectionsList');
  listEl.innerHTML = '';
  for (const ann of annotations) {
    if (visibleCategories[ann.category] === false) continue;
    const line = DetectionFormatter.formatLine(ann);
    const item = document.createElement('div');
    item.className = `det-line category-${ann.category}`;
    item.textContent = line;
    listEl.appendChild(item);
  }
}

function changePage(delta) {
  if (!selectedDoc) return;
  selectedPageIndex += delta;
  renderCurrentPage();
}

function updateCategoryVisibility(category, visible) {
    visibleCategories[category] = visible;
    
    // Redraw all canvases
    for (const [pageId, pageData] of Object.entries(pageDataMap)) {
        const canvas = document.getElementById(`canvas-${pageId}`);
        if (canvas && canvas.width > 0 && canvas.height > 0) {
            // Clear canvas
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // Redraw with updated visibility
            drawAnnotations(canvas, pageData.annotations, pageData.page_size);
        }
    }
}

function handleDownloadJson() {
    if (!currentDocId) return;
    window.open(`${API_BASE}/download/json/${currentDocId}`, '_blank');
}

function handleDownloadZip() {
    if (!currentDocId) return;
    window.open(`${API_BASE}/download/zip/${currentDocId}`, '_blank');
}

function showError(message) {
    errorMessage.textContent = `Error: ${message}`;
    errorMessage.classList.remove('hidden');
}

