class KoreanTextDecoder {
    constructor() {
        // Neural network architecture parameters
        this.inputSize = 11172; // Total possible Hangul syllables
        this.hiddenSize = 512;
        this.outputSize = 11172;
        
        // Initialize neural network weights (pre-trained)
        this.weights = this.loadPretrainedWeights();
        
        // Hangul Unicode ranges
        this.HANGUL_START = 0xAC00;
        this.HANGUL_END = 0xD7A3;
        
        // Jamo arrays for decomposition
        this.CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
        this.JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'];
        this.JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];

        // Initialize phonetic similarity matrix
        this.similarityMatrix = this.buildSimilarityMatrix();
        
        // Load embeddings for contextual analysis
        this.embeddings = this.loadEmbeddings();
        
        // Statistics tracking
        this.stats = {
            totalDecoded: 0,
            successfulDecodes: 0,
            failedDecodes: 0
        };
    }

    // Load pre-trained neural network weights
    loadPretrainedWeights() {
        // This would normally load from a file, but for demonstration we'll use a simplified version
        return {
            inputToHidden: new Float32Array(this.inputSize * this.hiddenSize),
            hiddenToOutput: new Float32Array(this.hiddenSize * this.outputSize),
            // Add pre-computed weights here
        };
    }

    // Build phonetic similarity matrix
    buildSimilarityMatrix() {
        const matrix = new Map();
        
        // Common phonetic similarities in Korean
        const similarities = [
            ['ㅐ', 'ㅔ', 0.9],
            ['ㅚ', 'ㅙ', 0.8],
            ['ㄱ', 'ㅋ', 0.7],
            ['ㄷ', 'ㅌ', 0.7],
            ['ㅂ', 'ㅍ', 0.7],
            ['ㅈ', 'ㅊ', 0.7],
            ['ㅜ', 'ㅡ', 0.6],
            // Add more phonetic similarities
        ];

        similarities.forEach(([char1, char2, score]) => {
            if (!matrix.has(char1)) matrix.set(char1, new Map());
            if (!matrix.has(char2)) matrix.set(char2, new Map());
            matrix.get(char1).set(char2, score);
            matrix.get(char2).set(char1, score);
        });

        return matrix;
    }

    // Load word embeddings for contextual analysis
    loadEmbeddings() {
        // Simplified word embeddings (would normally load from a file)
        return new Map([
            ['외국인', new Float32Array([0.2, 0.3, -0.1, 0.5])],
            ['방문', new Float32Array([0.1, 0.4, 0.2, -0.3])],
            // Add more embeddings
        ]);
    }

    // Convert character to one-hot encoding
    oneHotEncode(char) {
        const vector = new Float32Array(this.inputSize).fill(0);
        const code = char.charCodeAt(0) - this.HANGUL_START;
        if (code >= 0 && code < this.inputSize) {
            vector[code] = 1;
        }
        return vector;
    }

    // Neural network activation function (ReLU)
    relu(x) {
        return Math.max(0, x);
    }

    // Softmax function for output probabilities
    softmax(arr) {
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b);
        return exp.map(x => x / sum);
    }

    // Forward pass through the neural network
    forward(input) {
        // Convert input to one-hot encoding
        const inputVector = this.oneHotEncode(input);
        
        // Hidden layer
        const hidden = new Float32Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = 0;
            for (let j = 0; j < this.inputSize; j++) {
                sum += inputVector[j] * this.weights.inputToHidden[i * this.inputSize + j];
            }
            hidden[i] = this.relu(sum);
        }
        
        // Output layer
        const output = new Float32Array(this.outputSize);
        for (let i = 0; i < this.outputSize; i++) {
            let sum = 0;
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden[j] * this.weights.hiddenToOutput[i * this.hiddenSize + j];
            }
            output[i] = sum;
        }
        
        return this.softmax(output);
    }

    // Calculate phonetic similarity between characters
    calculatePhoneticSimilarity(char1, char2) {
        const decomp1 = this.decomposeHangul(char1);
        const decomp2 = this.decomposeHangul(char2);
        
        let similarity = 0;
        let count = 0;
        
        // Compare each component
        ['cho', 'jung', 'jong'].forEach(part => {
            if (decomp1[part] && decomp2[part]) {
                const score = this.similarityMatrix.get(decomp1[part])?.get(decomp2[part]) || 0;
                similarity += score;
                count++;
            }
        });
        
        return count > 0 ? similarity / count : 0;
    }

    // Decode a single character using the neural network
    decodeCharacter(char) {
        // Get network predictions
        const predictions = this.forward(char);
        
        // Find the most likely character
        let maxIndex = 0;
        let maxProb = predictions[0];
        
        for (let i = 1; i < predictions.length; i++) {
            if (predictions[i] > maxProb) {
                maxProb = predictions[i];
                maxIndex = i;
            }
        }
        
        // Convert back to character
        return String.fromCharCode(this.HANGUL_START + maxIndex);
    }

    // Main decode function with context awareness
    async decode(text, options = {}) {
        const {
            contextWindow = 2,
            threshold = 0.7,
            includeDecomposition = false
        } = options;

        try {
            this.stats.totalDecoded++;
            
            // Split text into words
            const words = text.split(/\s+/);
            const decodedWords = [];
            
            for (let i = 0; i < words.length; i++) {
                const word = words[i];
                const chars = Array.from(word);
                const decodedChars = [];
                
                // Process each character with context
                for (let j = 0; j < chars.length; j++) {
                    const char = chars[j];
                    const decoded = this.decodeCharacter(char);
                    
                    // Check context for better accuracy
                    const context = this.getContext(words, i, contextWindow);
                    const contextScore = this.evaluateContext(decoded, context);
                    
                    decodedChars.push(contextScore > threshold ? decoded : char);
                }
                
                decodedWords.push(decodedChars.join(''));
            }
            
            const decodedText = decodedWords.join(' ');
            const success = decodedText !== text;
            
            // Update statistics
            if (success) {
                this.stats.successfulDecodes++;
            } else {
                this.stats.failedDecodes++;
            }
            
            // Prepare response
            const response = {
                original: text,
                decoded: decodedText,
                success
            };
            
            if (includeDecomposition) {
                response.decomposition = this.decomposeText(text);
            }
            
            return response;

        } catch (error) {
            console.error('Decoding error:', error);
            this.stats.failedDecodes++;
            throw new Error(`Decoding failed: ${error.message}`);
        }
    }

    // Get context around a word
    getContext(words, currentIndex, windowSize) {
        const start = Math.max(0, currentIndex - windowSize);
        const end = Math.min(words.length, currentIndex + windowSize + 1);
        return words.slice(start, end);
    }

    // Evaluate context using word embeddings
    evaluateContext(char, context) {
        if (!this.embeddings.has(char)) return 0;
        
        const charEmbedding = this.embeddings.get(char);
        let score = 0;
        let count = 0;
        
        context.forEach(word => {
            if (this.embeddings.has(word)) {
                const wordEmbedding = this.embeddings.get(word);
                score += this.cosineSimilarity(charEmbedding, wordEmbedding);
                count++;
            }
        });
        
        return count > 0 ? score / count : 0;
    }

    // Cosine similarity between vectors
    cosineSimilarity(vec1, vec2) {
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        
        for (let i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    // Decompose text into Jamo components
    decomposeText(text) {
        return Array.from(text).map(char => this.decomposeHangul(char));
    }

    // Decompose a single Hangul character
    decomposeHangul(char) {
        const code = char.charCodeAt(0);
        
        if (code >= this.HANGUL_START && code <= this.HANGUL_END) {
            const offset = code - this.HANGUL_START;
            
            const jong = offset % 28;
            const jung = ((offset - jong) / 28) % 21;
            const cho = (((offset - jong) / 28) - jung) / 21;

            return {
                cho: this.CHOSUNG[cho],
                jung: this.JUNGSUNG[jung],
                jong: this.JONGSUNG[jong],
                original: char
            };
        }
        
        return {
            cho: '',
            jung: '',
            jong: '',
            original: char
        };
    }

    // Get statistics
    getStatistics() {
        return { ...this.stats };
    }

    // Reset statistics
    resetStatistics() {
        this.stats = {
            totalDecoded: 0,
            successfulDecodes: 0,
            failedDecodes: 0
        };
    }
}

// Export the decoder class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KoreanTextDecoder;
} else {
    window.KoreanTextDecoder = KoreanTextDecoder;
}