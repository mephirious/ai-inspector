// Handle multi-file upload and POST to /detect

window.FileUploadHandler = (function () {
  const API_BASE = window.location.origin;

  async function uploadFiles(files) {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    const resp = await fetch(`${API_BASE}/detect`, {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Upload failed");
    }
    return resp.json();
  }

  return { uploadFiles };
})();


