/**
 * Analisador de Sentimentos com TensorFlow.js
 * Usa um modelo próprio treinado com dados de reviews
 */

class SentimentAnalyzer {
    constructor() {
        this.model = null;
        this.wordIndex = null;
        this.maxLen = 100;
        this.vocabSize = 2000;

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadOrTrainModel();
        this.hideLoading();
    }

    setupEventListeners() {
        document.getElementById('analyzeBtn').addEventListener('click', () => this.analyze());

        document.getElementById('textInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.analyze();
            }
        });

        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.getElementById('textInput').value = btn.dataset.text;
                this.analyze();
            });
        });
    }

    updateLoadingText(text) {
        document.getElementById('loadingText').textContent = text;
    }

    async loadOrTrainModel() {
        const MODEL_VERSION = 'v1';
        const savedVersion = localStorage.getItem('sentiment-model-version');

        if (savedVersion === MODEL_VERSION) {
            try {
                this.model = await tf.loadLayersModel('localstorage://sentiment-model');
                this.wordIndex = JSON.parse(localStorage.getItem('sentiment-word-index'));
                if (this.model && this.wordIndex) {
                    console.log('Modelo carregado do localStorage');
                    return;
                }
            } catch (e) {
                console.log('Erro ao carregar modelo:', e);
            }
        }

        await this.trainModel();
        localStorage.setItem('sentiment-model-version', MODEL_VERSION);
    }

    getTrainingData() {
        // Dataset de treinamento com frases em inglês
        const positiveTexts = [
            "I love this product it is amazing",
            "This is the best thing ever",
            "Absolutely wonderful experience",
            "Great quality and fast shipping",
            "Exceeded my expectations completely",
            "Perfect works exactly as described",
            "Highly recommend this to everyone",
            "Fantastic product great value",
            "So happy with my purchase",
            "Amazing customer service",
            "Best purchase I have ever made",
            "Incredible quality and design",
            "Love it works perfectly",
            "Excellent product will buy again",
            "Super fast delivery great product",
            "Very satisfied with this purchase",
            "Outstanding quality highly recommend",
            "This made my day so happy",
            "Wonderful product great price",
            "Impressed with the quality",
            "Five stars not enough",
            "Better than expected",
            "Beautiful product love it",
            "Great experience overall",
            "Very pleased with this",
            "Awesome product works great",
            "Perfect gift idea",
            "Really happy with this",
            "Good quality great value",
            "Amazing just amazing",
            "The best product ever",
            "I am very happy",
            "This is great",
            "Love this so much",
            "Fantastic experience",
            "Really good product",
            "Very nice quality",
            "Happy customer here",
            "Will definitely buy again",
            "Pleasantly surprised",
            "Such a great product",
            "Worth every penny",
            "Exactly what I needed",
            "Made me smile",
            "Brilliant product",
            "So good I bought two",
            "Top quality item",
            "Really impressed",
            "Loving this product",
            "Great find"
        ];

        const negativeTexts = [
            "This is terrible do not buy",
            "Worst purchase ever made",
            "Very disappointed with quality",
            "Broke after one day of use",
            "Complete waste of money",
            "Horrible customer service",
            "Never buying from here again",
            "Does not work as advertised",
            "Very poor quality product",
            "Absolutely awful experience",
            "Returned immediately",
            "Such a disappointment",
            "Not worth the money",
            "Terrible quality avoid",
            "Hate this product",
            "Worst thing I ever bought",
            "Really bad quality",
            "Do not recommend at all",
            "Very unhappy with this",
            "Total garbage",
            "Broken on arrival",
            "Scam do not buy",
            "Extremely disappointed",
            "Awful just awful",
            "Waste of time and money",
            "Very frustrated",
            "Bad experience overall",
            "Not as described",
            "Poor quality avoid",
            "Regret this purchase",
            "This is bad",
            "Very angry",
            "Hate it",
            "Terrible product",
            "Worst ever",
            "Not good at all",
            "Disappointed",
            "Bad quality",
            "Do not buy this",
            "Horrible product",
            "Very upset",
            "Cheap junk",
            "Does not work",
            "Useless product",
            "Angry customer",
            "Feels like a scam",
            "Not satisfied",
            "Unhappy with purchase",
            "Low quality",
            "Avoid this"
        ];

        return { positiveTexts, negativeTexts };
    }

    buildVocabulary(texts) {
        const wordCounts = {};

        texts.forEach(text => {
            const words = text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/);
            words.forEach(word => {
                if (word.length > 1) {
                    wordCounts[word] = (wordCounts[word] || 0) + 1;
                }
            });
        });

        // Ordenar por frequência e pegar as mais comuns
        const sortedWords = Object.entries(wordCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, this.vocabSize - 2);

        const wordIndex = { '<PAD>': 0, '<UNK>': 1 };
        sortedWords.forEach(([word], idx) => {
            wordIndex[word] = idx + 2;
        });

        return wordIndex;
    }

    textToSequence(text) {
        const words = text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/);
        const sequence = words.map(word => this.wordIndex[word] || 1); // 1 = <UNK>

        // Padding
        if (sequence.length < this.maxLen) {
            return [...new Array(this.maxLen - sequence.length).fill(0), ...sequence];
        }
        return sequence.slice(-this.maxLen);
    }

    async trainModel() {
        this.updateLoadingText('Preparando dados de treinamento...');

        const { positiveTexts, negativeTexts } = this.getTrainingData();
        const allTexts = [...positiveTexts, ...negativeTexts];

        // Construir vocabulário
        this.wordIndex = this.buildVocabulary(allTexts);
        localStorage.setItem('sentiment-word-index', JSON.stringify(this.wordIndex));

        // Converter textos para sequências
        const sequences = allTexts.map(text => this.textToSequence(text));
        const labels = [
            ...new Array(positiveTexts.length).fill(1),
            ...new Array(negativeTexts.length).fill(0)
        ];

        // Shuffle dados
        const indices = [...Array(sequences.length).keys()];
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const shuffledSequences = indices.map(i => sequences[i]);
        const shuffledLabels = indices.map(i => labels[i]);

        // Criar tensores
        const xTrain = tf.tensor2d(shuffledSequences, [shuffledSequences.length, this.maxLen]);
        const yTrain = tf.tensor2d(shuffledLabels.map(l => [l]), [shuffledLabels.length, 1]);

        this.updateLoadingText('Construindo modelo...');

        // Criar modelo
        this.model = tf.sequential();

        // Embedding layer
        this.model.add(tf.layers.embedding({
            inputDim: this.vocabSize,
            outputDim: 32,
            inputLength: this.maxLen
        }));

        // LSTM
        this.model.add(tf.layers.lstm({
            units: 32,
            returnSequences: false
        }));

        // Dense layers
        this.model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
        this.model.add(tf.layers.dropout({ rate: 0.3 }));
        this.model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        this.model.compile({
            optimizer: tf.train.adam(0.01),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        this.updateLoadingText('Treinando modelo...');

        // Treinar
        await this.model.fit(xTrain, yTrain, {
            epochs: 30,
            batchSize: 16,
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const acc = (logs.acc * 100).toFixed(0);
                    this.updateLoadingText(`Treinando... Época ${epoch + 1}/30 - Acurácia: ${acc}%`);
                }
            }
        });

        // Limpar tensores
        xTrain.dispose();
        yTrain.dispose();

        // Salvar modelo
        await this.model.save('localstorage://sentiment-model');
        console.log('Modelo treinado e salvo!');
    }

    async analyze() {
        const text = document.getElementById('textInput').value.trim();

        if (!text) {
            alert('Por favor, digite um texto para analisar.');
            return;
        }

        if (!this.model) {
            alert('Modelo ainda carregando...');
            return;
        }

        // Converter texto para sequência
        const sequence = this.textToSequence(text);
        const input = tf.tensor2d([sequence], [1, this.maxLen]);

        // Predição
        const prediction = await this.model.predict(input).data();
        input.dispose();

        const positiveScore = prediction[0];
        const negativeScore = 1 - positiveScore;

        this.updateUI(positiveScore, negativeScore);
    }

    updateUI(positiveScore, negativeScore) {
        const emoji = document.getElementById('emoji');
        const label = document.getElementById('sentimentLabel');
        const positiveBar = document.getElementById('positiveBar');
        const negativeBar = document.getElementById('negativeBar');
        const positiveValue = document.getElementById('positiveValue');
        const negativeValue = document.getElementById('negativeValue');

        // Atualizar barras
        const posPercent = (positiveScore * 100).toFixed(1);
        const negPercent = (negativeScore * 100).toFixed(1);

        positiveBar.style.width = `${posPercent}%`;
        negativeBar.style.width = `${negPercent}%`;
        positiveValue.textContent = `${posPercent}%`;
        negativeValue.textContent = `${negPercent}%`;

        // Atualizar emoji e label
        emoji.classList.remove('animate');
        void emoji.offsetWidth; // Trigger reflow
        emoji.classList.add('animate');

        label.classList.remove('positive', 'negative', 'neutral');

        if (positiveScore > 0.6) {
            emoji.textContent = '😊';
            label.textContent = 'Sentimento Positivo';
            label.classList.add('positive');
        } else if (positiveScore < 0.4) {
            emoji.textContent = '😠';
            label.textContent = 'Sentimento Negativo';
            label.classList.add('negative');
        } else {
            emoji.textContent = '😐';
            label.textContent = 'Sentimento Neutro';
            label.classList.add('neutral');
        }
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }
}

// Iniciar
document.addEventListener('DOMContentLoaded', () => {
    new SentimentAnalyzer();
});
