# Analisador de Sentimentos com TensorFlow.js

Aplicação web que utiliza Deep Learning para analisar o sentimento de textos em tempo real, classificando-os como positivo, negativo ou neutro.

## Demo

Digite qualquer texto em inglês e veja a IA classificar o sentimento instantaneamente!

## Como Funciona

1. **Pré-processamento**: O texto é tokenizado e convertido em sequência numérica
2. **Embedding**: Palavras são convertidas em vetores densos de 32 dimensões
3. **LSTM**: Rede neural recorrente processa a sequência capturando contexto
4. **Classificação**: Camada densa final retorna probabilidade de sentimento positivo

## Tecnologias

- **TensorFlow.js** - Machine Learning no navegador
- **LSTM (Long Short-Term Memory)** - Rede neural para sequências
- **Word Embeddings** - Representação vetorial de palavras
- **NLP (Natural Language Processing)** - Processamento de linguagem natural

## Arquitetura do Modelo

```
Input (sequência de 100 tokens)
    ↓
Embedding (vocab=2000, dim=32)
    ↓
LSTM (32 unidades)
    ↓
Dense (16) + ReLU
    ↓
Dropout (0.3)
    ↓
Dense (1) + Sigmoid
    ↓
Output (probabilidade 0-1)
```

## Dataset de Treinamento

O modelo é treinado com 100 frases de exemplo (50 positivas + 50 negativas) diretamente no navegador. O treinamento leva cerca de 10-20 segundos.

## Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/sentiment-analyzer.git
cd sentiment-analyzer
```

2. Inicie um servidor local:
```bash
# Com Python
python -m http.server 8000

# Ou com Node.js
npx serve
```

3. Abra no navegador: `http://localhost:8000`


## Conceitos de Machine Learning Demonstrados

- **Word Embeddings**: Representação semântica de palavras
- **Redes Neurais Recorrentes (LSTM)**: Processamento de sequências
- **Transfer Learning**: Modelo treinável no browser
- **NLP**: Tokenização, padding, vocabulário
- **Classificação Binária**: Sigmoid + Binary Cross-Entropy

## Possíveis Melhorias

- Usar modelo pré-treinado maior (Universal Sentence Encoder)
- Adicionar suporte a português
- Implementar análise de aspectos
- Salvar histórico de análises

## Autor

Iago Machado

## Licença

MIT
