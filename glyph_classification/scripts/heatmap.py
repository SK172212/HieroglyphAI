import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("glyph_classification/models/v2/resnet18_v2_frozenSplit_min6_12ep_rot10_confusion_matrix.csv")

# remove the label column
df = df.set_index("true\\pred")

plt.figure(figsize=(12,10))

sns.heatmap(
    df,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True
)

plt.title("Classification Confusion Matrix (Version 2)", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig("confusion_matrix_v2.png", dpi=300)
plt.show()