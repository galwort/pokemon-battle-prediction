import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

pokemon = pd.read_csv("pokemon_battles/pokemon.csv")
sns.barplot(x="Type 1", y="HP", data=pokemon)
# plt.show()

battles = pd.read_csv("pokemon_battles/combats.csv")

poke_battles = battles.merge(pokemon, left_on="First_pokemon", right_on="#")
poke_battles = poke_battles.merge(pokemon, left_on="Second_pokemon", right_on="#")
poke_battles['Target'] = poke_battles['Winner'] == poke_battles['First_pokemon']

poke_battles = poke_battles.drop(['#_x', 'Name_x', '#_y', 'Name_y'], axis=1)
poke_battles.Type_1_x = poke_battles["Type 1_x"].astype('category')
poke_battles.Type_2_x = poke_battles["Type 2_x"].astype('category')
poke_battles.Type_1_y = poke_battles["Type 1_y"].astype('category')
poke_battles.Type_2_y = poke_battles["Type 2_y"].astype('category')

poke_battles = pd.get_dummies(poke_battles, columns=['Type 1_x', 'Type 2_x', 'Type 1_y', 'Type 2_y'])

X = poke_battles.drop(['Winner', 'Target'], axis=1)
y = poke_battles.Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(accuracy_score(y_test, y_pred))
