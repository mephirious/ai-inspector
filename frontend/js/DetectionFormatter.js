// Format detection lines for right panel
// Example line:
// "annotation_321 label_54 91.5% (602, 708) 119×81"

window.DetectionFormatter = {
  formatLine: function (ann) {
    const id = ann.id;
    const category = ann.category;
    const scorePct = ann.score != null ? (ann.score * 100).toFixed(1) + "%" : "";
    const x = Math.round(ann.bbox.x);
    const y = Math.round(ann.bbox.y);
    const w = Math.round(ann.bbox.width);
    const h = Math.round(ann.bbox.height);
    return `${id} ${category} ${scorePct} (${x}, ${y}) ${w}×${h}`;
  },
};


