# NOTAS — artigo Neural IOM (ler isto primeiro)

Atualizado: 2026-11-25

## Tese (decidida pelo autor)
- Rede = matriz insumo-produto com ativação Cobb-Douglas; elasticidade exata porque o modelo é restrito (essa é a ideia, não um defeito).
- Posicionamento: estimador estruturado interpretável por construção. NÃO é aproximador universal (rede inteira = uma Cobb-Douglas; ver Prop. 1 no Paper.tex).
- Tom moderado: sem "paradigm shift" / "resolve o black-box".

## Estado do Paper.tex
- Reescritos: título, resumo, introdução, teoria (Prop. colapso + Prop. Leontief, Teorema 2 corrigido), simulação (parâmetros conforme código), Solow, discussão, conclusão.
- Tabelas de resultados: números NÃO alterados.

## Pendências (buscar `TODO(author)` no Paper.tex)
1. Solow: scripts dos Exp. 1 e 3 usam `sum_equal_one=1` (softmax, sem coef. negativo), mas as Tabelas 8 e 10 têm negativos. Só o Exp. 2 usa 0. Confirmar qual config gerou cada tabela.
2. Solow: `s` no código = `rnna` (estoque de capital), não razão investimento/produto.
3. g+δ: 0,05 nos scripts 1–2; 0,07 no script 3.
4. Tabelas Spearman e Top-K Rank têm células idênticas: conferir nos CSVs.
5. Simulações sem baselines OLS log-log e gradiente médio da FNN em logs; sem testes pareados. Solow: 1 rodada por especificação, sem IC.
6. Compilar com svjour3 + sample.bib (só testei com classe substituta).

## Parâmetros reais (do código)
- Sim: n=10.000; split 64/16/20; lr=1e-5; Adam; 1.000 épocas; patience=5; NAS 10 candidatas × 500 épocas; softmax ligado.
- "Verdade" = participação de cada variável no resultado (elasticidade nas médias no caso linear), não o β.
- Solow: corte transversal por país (média 1971–2019 de ln Y/L; n e s de 1960–1985); avaliação dentro da amostra; 30 candidatas × 1.000 épocas, depois 10.000; lr 1e-5 (FNN) e 1e-4 (Cobb).
- Neural IOM empata com OLS em R² nas especificações 1 e 3; OLS melhor na 2.

## Conferências (decisão pendente do autor)
- RES: Sessão Geral fecha 18/01/2027 (artigo completo); Speed/Poster 5/04/2027. Resumo 500 palavras. Sem taxa de submissão. Página mistura 2026/2027: confirmar com events@res.org.uk.
- IAAE: 2026 foi na Nova SBE (Portugal); prazo de 2027 NÃO verificado.
- Plano sugerido: resolver pendências até meados de jan → RES Geral; senão Speed em abril. Pode-se enviar 1 Geral + 1 Speed/Poster.
- Não verificado: se apresentar na RES impede depois publicar em periódico.
