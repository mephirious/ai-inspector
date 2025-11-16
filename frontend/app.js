// Global state
let currentDocId = null;
let currentResults = null;
let pageDataMap = {}; // Store page data for redrawing
let visibleCategories = {
    signature: true,
    stamp: true,
    qr: true
};

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');
const pagesContainer = document.getElementById('pagesContainer');
const errorMessage = document.getElementById('errorMessage');
const toggleSignature = document.getElementById('toggleSignature');
const toggleStamp = document.getElementById('toggleStamp');
const toggleQR = document.getElementById('toggleQR');
const downloadJsonBtn = document.getElementById('downloadJsonBtn');
const downloadZipBtn = document.getElementById('downloadZipBtn');

// API base URL
const API_BASE = window.location.origin;

// Event listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
uploadArea.addEventListener('dragleave', handleDragLeave);
fileInput.addEventListener('change', handleFileSelect);
uploadBtn.addEventListener('click', handleUpload);
toggleSignature.addEventListener('change', () => updateCategoryVisibility('signature', toggleSignature.checked));
toggleStamp.addEventListener('change', () => updateCategoryVisibility('stamp', toggleStamp.checked));
toggleQR.addEventListener('change', () => updateCategoryVisibility('qr', toggleQR.checked));
downloadJsonBtn.addEventListener('click', handleDownloadJson);
downloadZipBtn.addEventListener('click', handleDownloadZip);

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
        const fileName = fileInput.files[0].name;
        uploadArea.querySelector('.upload-text').textContent = `Selected: ${fileName}`;
    } else {
        uploadBtn.disabled = true;
        uploadArea.classList.remove('has-file');
    }
}

async function handleUpload() {
    if (!fileInput.files.length) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    // Show loading
    loadingIndicator.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorMessage.classList.add('hidden');
    uploadBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/detect`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        currentDocId = data.document_id;
        currentResults = data.results;

        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        loadingIndicator.classList.add('hidden');
        uploadBtn.disabled = false;
    }
}

function displayResults(data) {
    pagesContainer.innerHTML = '';
    pageDataMap = {}; // Reset page data map
    
    // Get first document (should be only one)
    const docName = Object.keys(data.results)[0];
    const pages = data.results[docName];

    for (const [pageId, pageData] of Object.entries(pages)) {
        pageDataMap[pageId] = pageData; // Store for redrawing
        const pageDiv = createPageElement(pageId, pageData, docName);
        pagesContainer.appendChild(pageDiv);
    }

    resultsSection.classList.remove('hidden');
}

function createPageElement(pageId, pageData, docName) {
    const pageDiv = document.createElement('div');
    pageDiv.className = 'page-container';
    pageDiv.id = `page-${pageId}`;

    const pageHeader = document.createElement('div');
    pageHeader.className = 'page-header';
    pageHeader.innerHTML = `
        <h3>${pageId.replace('_', ' ').toUpperCase()}</h3>
        <p class="page-info">Size: ${pageData.page_size.width} × ${pageData.page_size.height}px | 
        Annotations: ${Object.keys(pageData.annotations).length}</p>
    `;

    const canvasContainer = document.createElement('div');
    canvasContainer.className = 'canvas-container';
    
    // Create image element to load original image
    const img = document.createElement('img');
    img.id = `img-${pageId}`;
    img.src = `${API_BASE}/images/${currentDocId}/${pageId}?original=true`;
    img.alt = `${pageId}`;
    img.style.maxWidth = '100%';
    img.style.height = 'auto';
    img.style.border = '1px solid #ddd';
    img.style.borderRadius = '4px';
    img.style.display = 'block';
    
    // Create canvas for overlay annotations (for toggling)
    const canvas = document.createElement('canvas');
    canvas.id = `canvas-${pageId}`;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.pointerEvents = 'none';
    
    // Store page data on canvas element for redrawing
    canvas.dataset.pageId = pageId;
    
    // Wait for image to load, then set canvas size and draw
    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        drawAnnotations(canvas, pageData.annotations, pageData.page_size);
    };
    
    const imageWrapper = document.createElement('div');
    imageWrapper.style.position = 'relative';
    imageWrapper.style.display = 'inline-block';
    imageWrapper.appendChild(img);
    imageWrapper.appendChild(canvas);
    
    canvasContainer.appendChild(imageWrapper);
    
    const annotationsList = document.createElement('div');
    annotationsList.className = 'annotations-list';
    annotationsList.innerHTML = '<h4>Detections:</h4>';
    
    for (const [annId, ann] of Object.entries(pageData.annotations)) {
        const annItem = document.createElement('div');
        annItem.className = `annotation-item category-${ann.category}`;
        annItem.innerHTML = `
            <span class="ann-id">${annId}</span>
            <span class="ann-category">${ann.category}</span>
            <span class="ann-confidence">${(ann.confidence * 100).toFixed(1)}%</span>
            <span class="ann-bbox">(${ann.bbox.x.toFixed(0)}, ${ann.bbox.y.toFixed(0)}) ${ann.bbox.width.toFixed(0)}×${ann.bbox.height.toFixed(0)}</span>
        `;
        annotationsList.appendChild(annItem);
    }

    pageDiv.appendChild(pageHeader);
    pageDiv.appendChild(canvasContainer);
    pageDiv.appendChild(annotationsList);

    return pageDiv;
}

function drawAnnotations(canvas, annotations, pageSize) {
    const ctx = canvas.getContext('2d');
    
    // Colors for categories
    const colors = {
        signature: '#0000FF', // blue
        stamp: '#FF0000',     // red
        qr: '#00FF00'         // green
    };

    // Draw all annotations
    for (const [annId, ann] of Object.entries(annotations)) {
        if (!visibleCategories[ann.category]) continue;

        const color = colors[ann.category] || '#FFFFFF';
        const bbox = ann.bbox;

        // Draw rectangle
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

        // Draw label
        ctx.fillStyle = color;
        ctx.font = '12px Arial';
        const label = `${ann.category} ${(ann.confidence * 100).toFixed(1)}%`;
        const labelY = Math.max(bbox.y - 5, 12);
        ctx.fillText(label, bbox.x, labelY);
    }
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

