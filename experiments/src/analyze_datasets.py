"""
Script para analisar e comparar datasets de treino e validação
para entender variações de performance entre modelos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_text_characteristics(df, dataset_name):
    """Analisa características textuais do dataset"""
    print(f"\n=== ANÁLISE TEXTUAL - {dataset_name.upper()} ===")
    
    # Comprimento dos textos
    text_lengths = df['text'].str.len()
    clean_lengths = df['text_clean'].str.len()
    
    print(f"Comprimento médio do texto original: {text_lengths.mean():.1f} caracteres")
    print(f"Comprimento médio do texto limpo: {clean_lengths.mean():.1f} caracteres")
    print(f"Redução média: {((text_lengths - clean_lengths) / text_lengths * 100).mean():.1f}%")
    
    # Vocabuário único
    all_words = ' '.join(df['text_clean'].dropna()).split()
    unique_words = len(set(all_words))
    total_words = len(all_words)
    
    print(f"Total de palavras: {total_words:,}")
    print(f"Palavras únicas: {unique_words:,}")
    print(f"Riqueza vocabular: {unique_words/total_words:.3f}")
    
    return {
        'avg_text_length': text_lengths.mean(),
        'avg_clean_length': clean_lengths.mean(),
        'vocab_richness': unique_words/total_words,
        'total_words': total_words,
        'unique_words': unique_words
    }

def analyze_entity_distribution(df, dataset_name):
    """Analisa distribuição por entidades"""
    print(f"\n=== ANÁLISE POR ENTIDADE - {dataset_name.upper()} ===")
    
    entity_sentiment = df.groupby(['entity', 'sentiment']).size().unstack(fill_value=0)
    entity_counts = df['entity'].value_counts()
    
    print(f"Número de entidades únicas: {len(entity_counts)}")
    print(f"Entidade mais comum: {entity_counts.index[0]} ({entity_counts.iloc[0]} tweets)")
    print(f"Entidade menos comum: {entity_counts.index[-1]} ({entity_counts.iloc[-1]} tweets)")
    
    # Análise de sentimento por entidade
    entity_sentiment_pct = entity_sentiment.div(entity_sentiment.sum(axis=1), axis=0) * 100
    
    print("\nTop 5 entidades com mais tweets positivos:")
    positive_pct = entity_sentiment_pct['Positive'].sort_values(ascending=False).head()
    for entity, pct in positive_pct.items():
        count = entity_sentiment.loc[entity, 'Positive'] if 'Positive' in entity_sentiment.columns else 0
        print(f"  {entity}: {pct:.1f}% ({count} tweets)")
    
    return entity_sentiment, entity_counts

def compare_model_performance_context(train_df, val_df):
    """Compara contextos que podem explicar variações de performance"""
    print("\n=== COMPARAÇÃO DE CONTEXTO PARA PERFORMANCE ===")
    
    # 1. Distribuição de classes
    train_dist = train_df['sentiment'].value_counts(normalize=True).sort_index()
    val_dist = val_df['sentiment'].value_counts(normalize=True).sort_index()
    
    print("Distribuição de classes (treino vs validação):")
    for sentiment in train_dist.index:
        train_pct = train_dist[sentiment] * 100
        val_pct = val_dist[sentiment] * 100 if sentiment in val_dist.index else 0
        diff = abs(train_pct - val_pct)
        print(f"  {sentiment}: Treino {train_pct:.1f}% vs Val {val_pct:.1f}% (dif: {diff:.1f}%)")
    
    # 2. Sobreposição de entidades
    train_entities = set(train_df['entity'].unique())
    val_entities = set(val_df['entity'].unique())
    common_entities = train_entities.intersection(val_entities)
    
    print(f"\nSobreposição de entidades:")
    print(f"  Entidades no treino: {len(train_entities)}")
    print(f"  Entidades na validação: {len(val_entities)}")
    print(f"  Entidades em comum: {len(common_entities)}")
    print(f"  Sobreposição: {len(common_entities)/len(val_entities)*100:.1f}%")
    
    # 3. Análise de entidades exclusivas
    train_only = train_entities - val_entities
    val_only = val_entities - train_entities
    
    print(f"\nEntidades exclusivas do treino (top 10): {list(train_only)[:10]}")
    print(f"Entidades exclusivas da validação: {list(val_only)}")
    
    return {
        'common_entities': len(common_entities),
        'train_only_entities': len(train_only),
        'val_only_entities': len(val_only),
        'class_distribution_diff': abs(train_dist - val_dist).mean() * 100
    }

def analyze_text_complexity(df, dataset_name):
    """Analisa complexidade textual"""
    print(f"\n=== ANÁLISE DE COMPLEXIDADE - {dataset_name.upper()} ===")
    
    # Número de palavras por tweet
    word_counts = df['text_clean'].dropna().str.split().str.len()
    
    print(f"Média de palavras por tweet: {word_counts.mean():.1f}")
    print(f"Mediana de palavras por tweet: {word_counts.median():.1f}")
    print(f"Máximo de palavras: {word_counts.max()}")
    print(f"Mínimo de palavras: {word_counts.min()}")
    
    # Palavras mais frequentes por sentimento
    for sentiment in ['Positive', 'Negative', 'Neutral', 'Irrelevant']:
        if sentiment in df['sentiment'].values:
            sentiment_texts = df[df['sentiment'] == sentiment]['text_clean'].dropna()
            if len(sentiment_texts) > 0:
                words = ' '.join(sentiment_texts).split()
                common_words = Counter(words).most_common(10)
                print(f"\nTop 10 palavras em tweets {sentiment.lower()}:")
                for word, count in common_words:
                    print(f"  {word}: {count}")
    
    return word_counts.describe()

def main():
    """Função principal de análise"""
    print("🔍 ANÁLISE DETALHADA DOS DATASETS")
    print("="*50)
    
    # Carregar datasets
    try:
        train_df = pd.read_csv('processed_train.csv')
        val_df = pd.read_csv('processed_validation.csv')
        
        print(f"✅ Datasets carregados com sucesso!")
        print(f"   Treino: {len(train_df):,} amostras")
        print(f"   Validação: {len(val_df):,} amostras")
        
    except FileNotFoundError as e:
        print(f"❌ Erro ao carregar datasets: {e}")
        return
    
    # Análises
    print("\n" + "="*50)
    
    # 1. Análise textual
    train_text_stats = analyze_text_characteristics(train_df, "treino")
    val_text_stats = analyze_text_characteristics(val_df, "validação")
    
    # 2. Análise por entidade
    train_entity_sentiment, train_entity_counts = analyze_entity_distribution(train_df, "treino")
    val_entity_sentiment, val_entity_counts = analyze_entity_distribution(val_df, "validação")
    
    # 3. Comparação de contexto
    context_comparison = compare_model_performance_context(train_df, val_df)
    
    # 4. Análise de complexidade
    train_word_stats = analyze_text_complexity(train_df, "treino")
    val_word_stats = analyze_text_complexity(val_df, "validação")
    
    # Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO DAS DIFERENÇAS CHAVE")
    print("="*60)
    
    print(f"\n🔢 Tamanho dos datasets:")
    print(f"   Treino: {len(train_df):,} amostras")
    print(f"   Validação: {len(val_df):,} amostras")
    print(f"   Razão: 1:{len(train_df)/len(val_df):.0f}")
    
    print(f"\n📈 Características textuais:")
    print(f"   Comprimento médio treino: {train_text_stats['avg_clean_length']:.1f} chars")
    print(f"   Comprimento médio validação: {val_text_stats['avg_clean_length']:.1f} chars")
    print(f"   Riqueza vocabular treino: {train_text_stats['vocab_richness']:.3f}")
    print(f"   Riqueza vocabular validação: {val_text_stats['vocab_richness']:.3f}")
    
    print(f"\n🎯 Distribuição de classes:")
    print(f"   Diferença média: {context_comparison['class_distribution_diff']:.1f}%")
    
    print(f"\n🏢 Sobreposição de entidades:")
    print(f"   Entidades em comum: {context_comparison['common_entities']}")
    print(f"   Entidades exclusivas treino: {context_comparison['train_only_entities']}")
    print(f"   Entidades exclusivas validação: {context_comparison['val_only_entities']}")
    
    print(f"\n💡 POSSÍVEIS EXPLICAÇÕES PARA VARIAÇÕES DE PERFORMANCE:")
    print(f"   1. Diferenças no vocabulário e comprimento dos textos")
    print(f"   2. Distribuição desigual de classes entre datasets")
    print(f"   3. Entidades diferentes (domínios distintos)")
    print(f"   4. Tamanho significativamente diferente dos datasets")
    print(f"   5. Complexidade textual variável")
    
    print(f"\n✅ CONCLUSÃO:")
    print(f"   Os modelos provavelmente foram treinados em contextos diferentes,")
    print(f"   com datasets que têm características distintas. Isso explica as")
    print(f"   variações de performance (KNN 0.94+ vs LinearSVC 0.93+).")

if __name__ == "__main__":
    main()