// Sets the base URL for Marine Learning API calls before main.js executes.
// You can override this by adding a data-api-base attribute in index.html or by
// defining window.MARINE_API_URL earlier in the page.
(function configureMarineApi() {
    if (typeof window.MARINE_API_URL === 'string' && window.MARINE_API_URL.length) {
        return;
    }

    const script = document.currentScript;
    const apiBase = script && typeof script.dataset.apiBase === 'string'
        ? script.dataset.apiBase.trim()
        : '';

    if (apiBase) {
        window.MARINE_API_URL = apiBase;
        return;
    }

    window.MARINE_API_URL = '';
})();
