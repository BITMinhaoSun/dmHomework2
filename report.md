## Task 1: 关联规则挖掘
代码
```
# --------------------------- Task1 Functions ---------------------------
def perform_association_rule_mining():
    """Task1: Product Category Association Rule Mining"""
    print("\n=== Starting Task 1: Product Category Association Rule Mining ===")
    configure_chinese_font()
    output_dir = generate_output_directory('task1')
    # Data loading with error handling
    try:
        with open('processed_data/transactions.json', 'r', encoding='utf-8') as f:
            transactions = json.load(f)
    except FileNotFoundError:
        print("Error: Transaction data file not found. Ensure processed_data/transactions.json exists")
        return
    except Exception as e:
        print(f"Error loading transaction data: {e}")
        return
    # Data exploration
    category_freq = Counter(cat for trans in transactions for cat in trans)
    print(f"\nTotal unique categories: {len(category_freq)}")
    print("\nTop 5 frequent categories:")
    for cat, cnt in category_freq.most_common(5):
        print(f"- {cat}: {cnt} ({cnt / len(transactions) * 100:.2f}%)")

    # Data transformation
    te = TransactionEncoder()
    encoded_data = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    encoded_data.to_csv(f"{output_dir}/encoded_transactions.csv", index=False)

    # Frequent itemset mining
    frequent_itemsets = apriori(encoded_data, min_support=0.02, use_colnames=True, max_len=3)
    print(f"\nFound {len(frequent_itemsets)} frequent itemsets (support >= 0.02)")
    frequent_itemsets.to_csv(f"{output_dir}/frequent_itemsets.csv", index=False)

    # Rule generation with confidence threshold
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print(f"Generated {len(rules)} association rules (confidence >= 0.5)")
    rules.to_csv(f"{output_dir}/association_rules.csv", index=False)

    # Target category analysis (Electronics)
    target_category = "电子产品"
    electronics_rules = rules[
        rules.apply(lambda x: target_category in x['antecedents'] or target_category in x['consequents'], axis=1)
    ]
    print(f"\nElectronics-related rules: {len(electronics_rules)}")
    electronics_rules.nlargest(5, 'lift').to_csv(f"{output_dir}/top_electronics_rules.csv", index=False)

    # Visualizations
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(category_freq.values()), y=category_freq.keys(), palette="viridis")
    plt.title("Product Category Frequency Distribution")
    plt.savefig(f"{output_dir}/category_frequency.png", dpi=300)

    if not rules.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='support', y='confidence', data=rules, size='lift', hue='lift', palette="viridis")
        plt.title("Association Rules: Support vs Confidence")
        plt.savefig(f"{output_dir}/rules_scatter.png", dpi=300)

    # Detailed report generation
    with open(f"{output_dir}/report.md", "w", encoding="utf-8") as f:
        f.write("# Task1 Analysis Report\n")
        f.write(f"## Data Summary\n- Transactions: {len(transactions)}\n- Categories: {len(category_freq)}\n")
        f.write("\n## Key Findings\n- Top category: {} ({}%)".format(*category_freq.most_common(1)[0]))
        f.write(f"\n- Strongest rule: {rules.nlargest(1, 'lift')['rule'].values[0]}")

    print(f"Task1 completed. Results saved to {output_dir}")

```
电子产品相关：

| 规则 | 支持度 | 置信度 | 提升度 |
|------|--------|--------|--------|
| 电子产品 → 食品 | 0.2213 | 0.4568 | 0.9430 |
| 食品 → 电子产品 | 0.2213 | 0.4568 | 0.9430 |
| 电子产品 → 服装 | 0.2224 | 0.4592 | 0.9420 |
| 服装 → 电子产品 | 0.2224 | 0.4563 | 0.9420 |
| 家居 → 电子产品 | 0.1134 | 0.4516 | 0.9323 |
## Task 2: 支付方式与商品类别关联
```

# --------------------------- Task2 Functions ---------------------------
def analyze_payment_category_correlations():
    """Task2: Payment Method & Category Correlation Analysis"""
    print("\n=== Starting Task 2: Payment Method & Category Correlation Analysis ===")
    configure_chinese_font()
    output_dir = generate_output_directory('task2')

    # Load and process transaction details
    try:
        with open('processed_data/transaction_details.json', 'r', encoding='utf-8') as f:
            transactions = json.load(f)
    except Exception as e:
        print(f"Data load error: {e}")
        return

    # Extract payment-category pairs
    payment_data = []
    high_value_payments = []
    for trans in transactions:
        payment = trans.get("payment_method", "Unknown")
        categories = trans.get("categories", [])
        amount = trans.get("total_amount", 0)

        for cat in set(categories):
            payment_data.append([f"Payment_{payment}", f"Category_{cat}"])

        if amount > 5000:
            high_value_payments.append(payment)

    # High-value payment analysis
    hv_dist = Counter(high_value_payments)
    print(f"\nHigh-value transactions: {len(high_value_payments)}")
    print("Top payment methods for high-value purchases:")
    for method, cnt in hv_dist.most_common(3):
        print(f"- {method}: {cnt} ({cnt / len(high_value_payments) * 100:.2f}%)")

    # Apriori for payment-category rules
    te = TransactionEncoder()
    encoded_data = pd.DataFrame(te.fit_transform(payment_data), columns=te.columns_)
    frequent_itemsets = apriori(encoded_data, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    # Filter rule directions
    payment_to_cat = rules[
        rules['antecedents'].apply(lambda x: any(item.startswith('Payment_') for item in x))
    ]
    cat_to_payment = rules[
        rules['consequents'].apply(lambda x: any(item.startswith('Payment_') for item in x))
    ]

    # Visualizations
    plt.figure(figsize=(12, 6))
    sns.countplot(x=high_value_payments, palette="viridis")
    plt.title("High-Value Payment Method Distribution")
    plt.savefig(f"{output_dir}/hv_payment_dist.png", dpi=300)

    if not payment_to_cat.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='lift', y='rule', data=payment_to_cat.nlargest(5, 'lift'), palette="viridis")
        plt.title("Top Payment-to-Category Rules by Lift")
        plt.savefig(f"{output_dir}/top_payment_rules.png", dpi=300)

    # Report generation
    with open(f"{output_dir}/report.md", "w", encoding="utf-8") as f:
        f.write("# Payment-Category Correlation Report\n")
        f.write(
            f"## High-Value Insights\n- {hv_dist.most_common(1)[0][0]} used in {hv_dist.most_common(1)[0][1]}% of high-value transactions")
        f.write(f"\n## Strongest Rule: {rules.nlargest(1, 'lift')['rule'].values[0]}")

    print(f"Task2 completed. Results saved to {output_dir}")
```
支付方式对商品类别关联规则

| 规则 | 支持度 | 置信度 | 提升度 |
|------|--------|--------|--------|
| 现金 → 家居 | 0.0148 | 0.1031 | 1.0036 |
| 信用卡 → 家居 | 0.0147 | 0.1030 | 1.0023 |
| 支付宝 → 食品 | 0.0284 | 0.1987 | 1.0023 |
| 储蓄卡 → 服装 | 0.0285 | 0.1997 | 1.0012 |
| 储蓄卡 → 家居 | 0.0147 | 0.1028 | 1.0008 |
商品类别对支付方式关联规则

| 规则 | 支持度 | 置信度 | 提升度 |
|------|--------|--------|--------|
| 家居 → 现金 | 0.0148 | 0.1436 | 1.0036 |
| 家居 → 信用卡 | 0.0147 | 0.1433 | 1.0023 |
| 食品 → 支付宝 | 0.0284 | 0.1431 | 1.0023 |
| 玩具 → 银联 | 0.0118 | 0.1433 | 1.0017 |
| 玩具 → 信用卡 | 0.0117 | 0.1431 | 1.0014 |
### Task 3: 时间序列模式分析
```
# -------------------------- Task3 Functions ---------------------------
def analyze_time_series_patterns():
    """Task3: Time Series Pattern Mining"""
    print("\n=== Starting Task 3: Time Series Pattern Mining ===")
    configure_chinese_font()
    # Date parsing with error handling
    valid_data = []
    date_errors = 0
    for item in data:
        try:
            item['date_obj'] = datetime.strptime(item['date'], "%Y-%m-%d")
            valid_data.append(item)
        except:
            date_errors += 1
    print(f"Parsed {len(valid_data)} records, {date_errors} date parsing errors")

    # Seasonal analysis
    def get_seasonal_distribution(key, name):
        dist = Counter(item[key] for item in valid_data)
        print(f"\n{name} Distribution:")
        for k, v in sorted(dist.items()):
            print(f"- {k}: {v} ({v / len(valid_data) * 100:.2f}%)")
        return dist

    quarterly_dist = get_seasonal_distribution('quarter', "Quarterly Sales")
    monthly_dist = get_seasonal_distribution('month', "Monthly Sales")
    weekday_dist = get_seasonal_distribution('day_of_week', "Weekday Sales",
                                             mapping={0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat",
                                                      6: "Sun"})

    # Category-time analysis
    category_timing = defaultdict(lambda: defaultdict(int))
    for item in valid_data:
        for cat in item['categories']:
            category_timing[cat][item['month']] += 1

    top_categories = [cat for cat, _ in
                      Counter(cat for item in valid_data for cat in item['categories']).most_common(5)]

    # Visualizations
    plt.figure(figsize=(12, 6))
    sns.countplot(x='quarter', data=valid_data, palette="viridis")
    plt.title("Quarterly Transaction Distribution")
    plt.savefig(f"{output_dir}/quarterly_sales.png", dpi=300)

    plt.figure(figsize=(15, 8))
    for cat in top_categories:
        months = range(1, 13)
        counts = [category_timing[cat].get(m, 0) for m in months]
        plt.plot(months, counts, label=cat)
    plt.title("Monthly Sales by Top Categories")
    plt.legend()
    plt.savefig(f"{output_dir}/monthly_categories.png", dpi=300)

    # Report generation
    with open(f"{output_dir}/report.md", "w", encoding="utf-8") as f:
        f.write("# Time Series Analysis Report\n")
        f.write(f"## Data Quality\n- Valid records: {len(valid_data)}\n- Invalid dates: {date_errors}")
        f.write(
            f"\n## Peak Season\n- Q{max(quarterly_dist, key=quarterly_dist.get)} has {quarterly_dist[max(quarterly_dist, key=quarterly_dist.get)]} transactions")

    print(f"Task3 completed. Results saved to {output_dir}")


```

按季度的购买分布
| 季度 | 交易数量 | 百分比 |
|------|----------|--------|
| Q1 | 1575807 | 28.01% |
| Q2 | 1339744 | 23.82% |
| Q3 | 1354860 | 24.09% |
| Q4 | 1354589 | 24.08% |
常见时序购买模式

| 排名 | 模式 | 频次 |
|------|------|------|
| 1 | 家居 → 服装 | 1648 |
| 2 | 电子产品 → 玩具 | 1607 |
| 3 | 玩具 → 食品 | 1603 |
| 4 | 玩具 → 家居 | 1584 |
| 5 | 食品 → 家居 | 1391 |
### Task 4: 退款模式分析
```
# --------------------------- Task4 Functions ---------------------------
def analyze_refund_patterns():
    """Task4: Refund Pattern Analysis"""
    print("\n=== Starting Task 4: Refund Pattern Analysis ===")
    configure_chinese_font()
    output_dir = generate_output_directory
    # Data analysis
    df = pd.DataFrame(refund_data)
    print("\nRefund data summary:")
    print(df[["refund_amount", "refund_reason"]].describe())

    # Refund reason distribution
    reason_dist = df['refund_reason'].value_counts(normalize=True)
    print("\nTop refund reasons:")
    for reason, prop in reason_dist.items():
        print(f"- {reason}: {prop * 100:.2f}%")

    # Category-wise refund analysis
    category_refund = df.groupby('category')['refund_amount'].agg(['mean', 'count'])
    print("\nCategory refund statistics:")
    print(category_refund)

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(x='refund_reason', data=df, palette="viridis")
    plt.title("Refund Reason Distribution")
    plt.savefig(f"{output_dir}/refund_reasons.png", dpi=300)

    plt.figure(figsize=(12, 4))
    sns.boxplot(x='category', y='refund_amount', data=df, palette="viridis")
    plt.title("Refund Amount by Category")
    plt.savefig(f"{output_dir}/category_refund_boxplot.png", dpi=300)

    # Report generation
    with open(f"{output_dir}/report.md", "w", encoding="utf-8") as f:
        f.write("# Refund Pattern Analysis Report\n")
        f.write(f"## Key Insights\n- {reason_dist.idxmax()} is the most common reason ({reason_dist.max() * 100:.2f}%)")
        f.write(
            f"\n- {category_refund['mean'].idxmax()} has the highest average refund amount ({category_refund['mean'].max():.2f}￥)")

    print(f"Task4 completed. Results saved to {output_dir}")


```