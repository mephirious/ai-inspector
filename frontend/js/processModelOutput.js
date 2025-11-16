// Parse model output and prepare structures for UI consumption
// Input shape:
// {
//   "docName.pdf": {
//     "page_1": {
//       "page_size": { "width": W, "height": H },
//       "annotations": [ { "annotation_001": { ... } }, ... ]
//     }
//   },
//   ...
// }
// Output:
// {
//   documents: [docName, ...],
//   pagesByDoc: { docName: [ "page_1", "page_2", ... ] },
//   pageData: { docName: { pageId: { page_size, annotations: [ {id, category, bbox, area, score} ] } } }
// }

window.processModelOutput = function processModelOutput(resultsJson) {
  const documents = Object.keys(resultsJson);
  const pagesByDoc = {};
  const pageData = {};

  for (const docName of documents) {
    const pages = resultsJson[docName] || {};
    const pageIds = [];
    pageData[docName] = {};

    for (const [pageId, page] of Object.entries(pages)) {
      const annArray = Array.isArray(page.annotations) ? page.annotations : [];
      // Skip pages with no annotations
      if (annArray.length === 0) continue;

      // Normalize annotations into flat array with id
      const normalized = [];
      for (const entry of annArray) {
        const annId = Object.keys(entry)[0];
        const annVal = entry[annId];
        normalized.push({
          id: annId,
          category: annVal.category,
          bbox: annVal.bbox,
          area: annVal.area,
          score: annVal.score,
        });
      }

      if (normalized.length === 0) continue;
      pageIds.push(pageId);
      pageData[docName][pageId] = {
        page_size: page.page_size,
        annotations: normalized,
      };
    }

    pagesByDoc[docName] = pageIds;
  }

  return { documents, pagesByDoc, pageData };
};


