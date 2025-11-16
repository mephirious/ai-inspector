// Render original page image and draw detections overlay with scaling

window.PageRenderer = (function () {
  function setImageAndCanvas(imgEl, canvasEl, src, pageSize) {
    return new Promise((resolve, reject) => {
      imgEl.onload = () => {
        // Use natural size from server image; expect it matches pageSize
        canvasEl.width = imgEl.naturalWidth;
        canvasEl.height = imgEl.naturalHeight;
        resolve();
      };
      imgEl.onerror = reject;
      imgEl.src = src;
    });
  }

  function drawAnnotations(canvas, annotations, visibleCategories) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const colors = {
      signature: "#0000FF",
      stamp: "#FF0000",
      qr: "#00FF00",
    };

    for (const ann of annotations) {
      if (visibleCategories && visibleCategories[ann.category] === false) continue;
      const color = colors[ann.category] || "#FFFFFF";
      const { x, y, width, height } = ann.bbox;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      const label = `${ann.category} ${ann.score != null ? (ann.score * 100).toFixed(1) + "%" : ""}`;
      ctx.fillStyle = color;
      ctx.font = "12px Arial";
      const labelY = Math.max(y - 5, 12);
      ctx.fillText(label, x, labelY);
    }
  }

  return {
    setImageAndCanvas,
    drawAnnotations,
  };
})();


