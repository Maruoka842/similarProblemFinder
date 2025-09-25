import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// Force the library to fetch models from the Hugging Face Hub
env.allowLocalModels = false;

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const problemSearch = document.getElementById('problem-search');
    const searchSuggestions = document.getElementById('search-suggestions');
    const embeddingTypeSelectText = document.getElementById('embedding-type-select-text');
    const embeddingTypeSelectProblem = document.getElementById('embedding-type-select-problem');
    const resultsDiv = document.getElementById('results');
    const loadingIndicator = document.getElementById('loading-indicator');
    const modelStatus = document.getElementById('model-status');
    const textInput = document.getElementById('text-input');
    const textSearchBtn = document.getElementById('text-search-btn');
    const toggleTagsVisibility = document.getElementById('toggle-tags-visibility');
    const textSearchCard = document.getElementById('text-search-card');
    const problemSearchCard = document.getElementById('problem-search-card');
    let isModelReady = false;

    // --- State ---
    let allProblemsData = [];
    let allCodesData = [];
    let selectedItem = null; // Can be a problem or a code object
    let activeTags = new Set();
    let showTags = JSON.parse(localStorage.getItem('showTags') || 'true');

    // --- AI Model Manager (for text input) ---
    const AI = {
        task: 'feature-extraction',
        model: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
        quantized: true,
        pipelineInstance: null,
        async getPipeline(progress_callback = null) {
            if (this.pipelineInstance === null) {
                this.pipelineInstance = pipeline(this.task, this.model, { quantized: this.quantized, progress_callback });
            }
            return this.pipelineInstance;
        },
        async vectorize(text) {
            const pipeline = await this.getPipeline();
            const output = await pipeline(text, { pooling: 'mean', normalize: true });
            return output.data;
        }
    };

    // --- Core Functions ---
    function cosineSimilarity(vecA, vecB) {
        if (!vecA || !vecB) return 0;
        const dotProduct = vecA.reduce((acc, val, i) => acc + val * vecB[i], 0);
        const normA = Math.sqrt(vecA.reduce((acc, val) => acc + val * val, 0));
        const normB = Math.sqrt(vecB.reduce((acc, val) => acc + val * val, 0));
        if (normA === 0 || normB === 0) return 0;
        return dotProduct / (normA * normB);
    }

    function findSimilarProblems(sourceVector, embeddingType) {
        const similarities = allProblemsData
            .map(p => {
                const targetVector = p[embeddingType];
                const score = cosineSimilarity(sourceVector, targetVector);
                return { problem: p, score: score };
            })
            .filter(item => item.score > 0.5);

        similarities.sort((a, b) => b.score - a.score);
        return similarities.filter(item => item.score < 0.9999).slice(0, 15);
    }

    function findProblemsViaCodeSimilarity(sourceCodeVector) {
        // Step 1: Find similar codes by comparing the source code vector to all other code vectors.
        const similarCodes = allCodesData
            .map(code => {
                const score = cosineSimilarity(sourceCodeVector, code.embedding);
                return { code: code, score: score };
            })
            .filter(item => item.score > 0.6 && item.score < 0.9999); // Exclude self

        similarCodes.sort((a, b) => b.score - a.score);

        // Take more than 15 to have candidates for mapping
        const topSimilarCodes = similarCodes.slice(0, 30);

        // Step 2: Map the found similar codes back to their parent problems.
        const similarProblems = [];
        const foundProblemIds = new Set();

        for (const item of topSimilarCodes) {
            if (similarProblems.length >= 15) break;

            const matchedProblem = allProblemsData.find(p => p.shortest_code_filename === item.code.filename);
            
            if (matchedProblem && !foundProblemIds.has(matchedProblem.problem_id)) {
                foundProblemIds.add(matchedProblem.problem_id);
                // Use the code-similarity score for the problem ranking
                similarProblems.push({ problem: matchedProblem, score: item.score });
            }
        }
        
        return similarProblems;
    }

    // --- UI Rendering ---
    function renderProblemResults(similarProblems) {
        resultsDiv.innerHTML = '';
        if (similarProblems.length === 0) {
            resultsDiv.innerHTML = '<p class="text-center">類似する問題が見つかりませんでした。</p>';
            return;
        }

        let headerHtml = '';
        if (activeTags.size > 0) {
            headerHtml = `
                <div class="alert alert-info d-flex justify-content-between align-items-center">
                    <span><strong>適用中のフィルター:</strong> ${Array.from(activeTags).map(tag => `<span class="badge bg-primary me-1">${tag}</span>`).join('')}</span>
                    <button class="btn btn-sm btn-outline-secondary" id="clear-filters-btn">クリア</button>
                </div>
            `;
        }
        resultsDiv.innerHTML = headerHtml;

        const listGroup = document.createElement('ul');
        listGroup.className = 'list-group';
        similarProblems.forEach(item => {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-start';
            li.innerHTML = `
                <div class="ms-2 me-auto">
                    <div class="fw-bold">${item.problem.title}</div>
                    <a href="${item.problem.url}" target="_blank" class="text-muted">${item.problem.problem_id}</a>
                    <div class="tags-container">${showTags ? item.problem.tags.map(tag => `<a href="#" class="badge bg-secondary me-1 tag-clickable" data-tag="${tag}">${tag}</a>`).join('') : ''}</div>
                </div>
                <span class="badge bg-primary rounded-pill">${item.score.toFixed(4)}</span>
            `;
            listGroup.appendChild(li);
        });
        resultsDiv.appendChild(listGroup);
    }

    function showSuggestions(query) {
        searchSuggestions.innerHTML = '';
        if (query.length < 2) return;
        const lowerCaseQuery = query.toLowerCase();
        
        // Always search and suggest from problems data, regardless of mode
        const filteredProblems = allProblemsData.filter(p => 
            p.problem_id.toLowerCase().includes(lowerCaseQuery) || p.title.toLowerCase().includes(lowerCaseQuery)
        ).slice(0, 10);

        filteredProblems.forEach(p => {
            const item = document.createElement('a');
            item.href = '#';
            item.className = 'list-group-item list-group-item-action';
            item.textContent = `[${p.problem_id}] ${p.title}`;
            // We only need to store the problem_id, the logic will handle the rest
            item.dataset.itemId = p.problem_id;
            searchSuggestions.appendChild(item);
        });
    }

    // --- Event Handlers ---
    resultsDiv.addEventListener('click', (e) => {
        const target = e.target.closest('.tag-clickable, #clear-filters-btn');
        if (!target) return;
        e.preventDefault();
        if (target.id === 'clear-filters-btn') {
            activeTags.clear();
        } else {
            const clickedTag = target.dataset.tag;
            activeTags.has(clickedTag) ? activeTags.delete(clickedTag) : activeTags.add(clickedTag);
        }
        // This behavior might need to be re-evaluated, for now it just clears the search
        problemSearch.value = '';
        resultsDiv.innerHTML = '';
    });

    function handleSearch() {
        if (!selectedItem) return;
        loadingIndicator.style.display = 'block';
        resultsDiv.innerHTML = '';

        const searchMode = embeddingTypeSelectProblem.value;
        let sourceVector = null;

        // Step 1: Get the source vector based on the selected item and mode
        if (searchMode.startsWith('code')) {
            const codeFilename = selectedItem.shortest_code_filename;
            if (codeFilename) {
                const correspondingCode = allCodesData.find(c => c.filename === codeFilename);
                if (correspondingCode) {
                    sourceVector = correspondingCode.embedding;
                }
            } else {
                resultsDiv.innerHTML = `<p class="text-center text-danger">問題「${selectedItem.problem_id}」に紐付けられたコードファイルがありません。</p>`;
                loadingIndicator.style.display = 'none';
                return;
            }
        } else {
            sourceVector = selectedItem[searchMode];
        }

        if (!sourceVector) {
            resultsDiv.innerHTML = '<p class="text-center text-danger">基準となるベクトルデータが見つかりません。</p>';
            loadingIndicator.style.display = 'none';
            return;
        }

        // Step 2: Call the appropriate search function based on the mode
        setTimeout(() => {
            let similarProblems;
            if (searchMode.startsWith('code')) {
                similarProblems = findProblemsViaCodeSimilarity(sourceVector);
            } else {
                similarProblems = findSimilarProblems(sourceVector, searchMode);
            }
            renderProblemResults(similarProblems);
            loadingIndicator.style.display = 'none';
        }, 10);
    }

    problemSearch.addEventListener('input', () => showSuggestions(problemSearch.value));
    problemSearch.addEventListener('focusout', () => setTimeout(() => { searchSuggestions.innerHTML = ''; }, 200));
    problemSearch.addEventListener('focusin', () => showSuggestions(problemSearch.value));

    searchSuggestions.addEventListener('click', (e) => {
        e.preventDefault();
        const target = e.target.closest('.list-group-item-action');
        if (!target) return;

        const itemId = target.dataset.itemId;
        selectedItem = allProblemsData.find(p => p.problem_id === itemId);

        if (selectedItem) {
            problemSearch.value = `[${selectedItem.problem_id}] ${selectedItem.title}`;
            searchSuggestions.innerHTML = '';
            handleSearch();
        }
    });

    embeddingTypeSelectProblem.addEventListener('change', () => {
        // When the comparison basis changes, re-run the search if an item is selected.
        // This keeps the selected item in the search bar and updates the results.
        if (selectedItem) {
            handleSearch();
        }
    });

    textSearchBtn.addEventListener('click', async () => {
        const text = textInput.value;
        if (text.trim().length < 10) {
            alert('検索するには、より長いテキストを入力してください。');
            return;
        }
        loadingIndicator.style.display = 'block';
        resultsDiv.innerHTML = '';
        const vector = await AI.vectorize(text);
        const similarProblems = findSimilarProblems(vector, embeddingTypeSelectText.value);
        renderProblemResults(similarProblems);
        loadingIndicator.style.display = 'none';
    });

    // --- Initialization ---
    async function init() {
        try {
            // Load problems data
            const problemsManifestResponse = await fetch('data/problems_data_manifest.json');
            if (!problemsManifestResponse.ok) throw new Error('Failed to load problems_data_manifest.json.');
            const problemsManifest = await problemsManifestResponse.json();

            const problemsPartPromises = problemsManifest.map(filename =>
                fetch(`data/${filename}`).then(res => {
                    if (!res.ok) throw new Error(`Failed to load ${filename}.`);
                    return res.json();
                })
            );
            const problemsParts = await Promise.all(problemsPartPromises);
            allProblemsData = problemsParts.flat();
            console.log(`Successfully loaded ${allProblemsData.length} problems from ${problemsParts.length} parts.`);

            // Load codes data
            const codesManifestResponse = await fetch('data/codes_data_manifest.json');
            if (!codesManifestResponse.ok) {
                console.warn('Could not load codes_data_manifest.json. Code search will be disabled.');
            } else {
                const codesManifest = await codesManifestResponse.json();
                const codesPartPromises = codesManifest.map(filename =>
                    fetch(`data/${filename}`).then(res => {
                        if (!res.ok) throw new Error(`Failed to load ${filename}.`);
                        return res.json();
                    })
                );
                const codesParts = await Promise.all(codesPartPromises);
                allCodesData = codesParts.flat();
                console.log(`Successfully loaded ${allCodesData.length} code snippets from ${codesParts.length} parts.`);
            }

            problemSearch.disabled = false;
            embeddingTypeSelectProblem.dispatchEvent(new Event('change')); // Set initial placeholder

        } catch (error) {
            console.error(error);
            problemSearch.disabled = true;
            problemSearch.placeholder = 'データ読込失敗';
            return;
        }

        try {
            await AI.getPipeline((progress) => {
                if (progress.status === 'ready') {
                    modelStatus.className = 'alert alert-success';
                    modelStatus.innerHTML = '<strong>AIモデルの準備が完了しました。</strong> テキスト入力での検索が利用可能です。';
                    isModelReady = true;
                    textInput.disabled = false;
                    textSearchBtn.disabled = false;
                }
            });
        } catch (error) {
            console.error(error);
            modelStatus.className = 'alert alert-warning';
            modelStatus.innerHTML = '<strong>テキスト入力検索は現在利用できません。</strong>(AIモデルの読み込み失敗)';
        }
    }

    init();
});