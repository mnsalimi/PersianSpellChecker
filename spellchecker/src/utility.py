def calc_metric(tokens, tags, predicteds):
    precision = 0
    precision_total = 0
    recall = 0
    recall_total = 0
    f1 = 0
    print("tokens", tokens)
    print("tags", tags)
    print("predicteds", predicteds)
    for i in range(len(predicteds)):
        if predicteds[i] != tokens[i]:
            precision_total += 1
            if predicteds[i] == tags[i]:
                precision += 1
        if tokens[i] == tags[i]:
            recall_total += 1
            if tokens[i] == predicteds[i]:
                recall += 1
    try:
        precision /= precision_total
    except:
        precision = 1
    try:
        recall /= recall_total
    except:
        recall = 1
    f1 = round(
        float(
            2*precision*recall/(precision+recall)
        ),
        2
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

if __name__ == "__main__":
    tokens = "بحران بی آتی در بسیاری ار شهرهای ایران".split(" ")
    predicteds = "بحران بی آبی دز بسیازی ار شهرهای ایرات".split(" ")
    tags = "بحران بی آبی دز بسیاری ار شفرهای ایران".split(" ")
    print("tokens")
    print(tokens)
    print()
    print("tags")
    print(tags)
    print("pred")
    print(predicteds)
    print()
    f1 = calc_metric(tokens, tags, predicteds)
    print(f1)