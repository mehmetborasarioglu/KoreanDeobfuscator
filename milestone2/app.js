document.addEventListener('DOMContentLoaded', () => {
    // Initialize decoder
    const decoder = new KoreanTextDecoder();

    // Get DOM elements
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const decodeButton = document.getElementById('decodeButton');
    const clearButton = document.getElementById('clearButton');
    const copyButton = document.getElementById('copyButton');
    const outputSection = document.getElementById('outputSection');
    const decodeMethod = document.getElementById('decodeMethod');
    const themeToggle = document.getElementById('themeToggle');
    const decompositionDisplay = document.getElementById('decompositionDisplay');

    // Statistics elements
    const totalDecoded = document.getElementById('totalDecoded');
    const successfulDecodes = document.getElementById('successfulDecodes');
    const failedDecodes = document.getElementById('failedDecodes');

    // Theme handling
    let isDark = false;
    themeToggle.addEventListener('click', () => {
        isDark = !isDark;
        document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
        themeToggle.textContent = isDark ? '☀️' : '🌙';
    });

    // Input handling
    inputText.addEventListener('input', () => {
        decodeButton.disabled = !inputText.value.trim();
    });

    // Clear button handling
    clearButton.addEventListener('click', () => {
        inputText.value = '';
        outputText.value = '';
        decompositionDisplay.innerHTML = '';
        outputSection.classList.add('hidden');
        decodeButton.disabled = true;
    });

    // Copy button handling
    copyButton.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(outputText.value);
            copyButton.textContent = '✅';
            setTimeout(() => {
                copyButton.textContent = '📋';
            }, 1000);
        } catch (err) {
            console.error('Failed to copy text:', err);
        }
    });

    // Display decomposition
    function displayDecomposition(decomposition) {
        decompositionDisplay.innerHTML = '';
        decomposition.forEach(char => {
            if (char.cho || char.jung || char.jong) {
                const block = document.createElement('div');
                block.className = 'char-block';
                block.innerHTML = `
                    <div>Original: ${char.original}</div>
                    <div>초성: ${char.cho || '-'}</div>
                    <div>중성: ${char.jung || '-'}</div>
                    <div>종성: ${char.jong || '-'}</div>
                `;
                decompositionDisplay.appendChild(block);
            }
        });
    }

    // Update statistics display
    function updateStats(stats) {
        totalDecoded.textContent = stats.totalDecoded;
        successfulDecodes.textContent = stats.successfulDecodes;
        failedDecodes.textContent = stats.failedDecodes;
    }

    // Decode button handling
    decodeButton.addEventListener('click', async () => {
        const text = inputText.value.trim();
        if (!text) return;

        decodeButton.disabled = true;
        decodeButton.textContent = 'Decoding...';

        try {
            const result = await decoder.decode(text, {
                method: decodeMethod.value,
                includeDecomposition: true,
                statistics: true
            });

            outputText.value = result.decoded;
            outputSection.classList.remove('hidden');
            
            if (result.decomposition) {
                displayDecomposition(result.decomposition);
            }
            
            if (result.statistics) {
                updateStats(result.statistics);
            }
        } catch (error) {
            console.error('Decoding error:', error);
            outputText.value = 'Error decoding text';
        } finally {
            decodeButton.textContent = 'Decode';
            decodeButton.disabled = false;
        }
    });
});