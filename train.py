from src.breaking_bad.training import train

if __name__ == "__main__":
    result = train()
    print("Treinamento concluído. Métricas do teste temporal:")
    print(result["test_metrics"])
