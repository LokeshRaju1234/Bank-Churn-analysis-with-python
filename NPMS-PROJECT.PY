import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r'C:\Users\loki6\OneDrive\Documents\BANK-CHURN-PYTHON.csv')
print(df)

print(df.head())
print(df.shape)


missing_values = df.isnull().sum()
print("Missing values in the data:\n",missing_values)
print(df.dtypes)

print(df.isnull())
print(df.isnull().sum().sum())

print("\nCategorical Information:")
print(df.info())

#checking for the duplicates in the dataset
print("\nDuplicates: ",df.duplicated().sum())

#removing duplicates
df.drop_duplicates(inplace=True)

#getting certain number of rows
print("\nDataInfo: \n",df.head(15))

print(df.describe())

creditscore_uniq = df["CreditScore"].unique()
print("\n",creditscore_uniq)

age_uniq = df["Age"].unique()
print("\n",age_uniq)



sorted_by_EstimatedSalary = df.sort_values('EstimatedSalary',ascending=False)
print("EstimatedSalary is sorted: \n",sorted_by_EstimatedSalary)

sorted_by_Balance = df.sort_values('Balance',ascending=False)
print("Balance is sorted: \n",sorted_by_Balance)

#summary statistics
mean_Age = df["Age"].mean()
print("Mean of Age: ",mean_Age)
Max_tenure = df["Tenure"].max()
print("Maximum years the person is with bank: ",Max_tenure)
print("Variation of the data: ",df["CreditScore"].std())

#filtering the data of tenure to get min value other than 0
filtred_tenure = df[df["Tenure"] > 0]
Min_tenure = filtred_tenure["Tenure"].min()
print("Minimum years the person is with the bank except (0 years): ",Min_tenure)

Sum_Salary = df["EstimatedSalary"].sum()
conver = Sum_Salary.astype(int)
print("Sum of the Estimated Salary :",conver)

#Calculating the customers who leave the bank(churned)
churned = df[df["Exited"] == 1].sum() 
print("The total customers who have exited from the bank: \n",churned["Exited"])

plt.hist(churned["Exited"],color = "orange",edgecolor = "black")
plt.xlabel("Exited")
plt.title("The total customers who have exited from the bank")
plt.show()


#calculating the customers percentage who have exited from the bank
churned_percentage = churned["Exited"]/len(df) * 100 #which will give you  percentage
print(f"Percentage of the people who left the bank: {churned_percentage:.2f}%")

# Sample dataset: Top churned customers based on balance
top_customers = df.nlargest(10, "Balance")

plt.figure(figsize=(10,6))
sns.barplot(x=top_customers["Balance"], y=top_customers["Surname"], palette="coolwarm")

plt.xlabel("Account Balance", fontsize=12)
plt.ylabel("Customer Surname", fontsize=12)
plt.title("Top 10 Customers with Highest Account Balance", fontsize=14)
plt.grid(axis='x', linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


Active_members = df["IsActiveMember"].sum()
print("Total active members: ",Active_members)

ax = sns.countplot(x = "IsActiveMember",data = df,palette="viridis")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Total active(1) and inactive customers(0)")    
plt.show()

active_credit_card_users= df[df["HasCrCard"] == 1].count()
print("Active members who are uisng credit card: ",active_credit_card_users["HasCrCard"])

#percentage of members who are active
Percentage_active_members = Active_members / len(df) * 100
print(f"\nTotal active members in percentage: {Percentage_active_members:.2f}%")

#percentage of members who are inactive
percentage_inactivate = 100 - Percentage_active_members
print("Percentage of inactive members:",percentage_inactivate,end="%")
print()
credit_card_users = df["HasCrCard"].value_counts()
print(credit_card_users)

plt.hist(active_credit_card_users,color = "grey",edgecolor = "black")
plt.xlabel("credit card users")
plt.title("total credit card users")
plt.show()

#checking for outliers
sns.boxplot(x = df["CreditScore"])
plt.title("Boxplot for Credit Score")
plt.show()

#finding that maximum number of products purchased from the bank
Most_products_buyed = df["NumOfProducts"].max()
print("Maximum No of products buyed by customer: ",Most_products_buyed)

#finding that most number of customers who buyed the products
Products = df["NumOfProducts"].mode()[0]
print("Most of the customers buyed:",Products,"No of products")


#Finding out the number of products buyed according to age
age = df.groupby("Age")["NumOfProducts"].sum()
print("According to the age: \n",age)


plt.bar(age.index,age.values,color = "darkviolet",edgecolor = "black")
plt.xlabel("products")
plt.title("Total number of products buyed according to the age")
plt.show()

age1 = df[["Age","CreditScore","Surname","Gender"]].max()
print("\nPerson who is with highest creditScore: \n",age1)


#country wise exited customers
country = df.groupby("Geography")["Exited"].sum()
print("\n",country)
print()

colors = ["black","pink","blue"]
ex = [0,0,0.1]
plt.pie(country,labels = country.index,autopct="%.2f",colors=colors,explode = ex)
plt.title("Country wise exited customers")
plt.show()

#genderwise exited customers
gender_exit = df.groupby("Gender")["Exited"].sum()

sns.barplot(x=gender_exit.index, y=gender_exit.values, palette="pastel")
plt.title("Total Number of Customers Who Exited by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Exited Customers")
plt.show()

print()
#in which country the products is getting selled
c1 = df[["Geography","NumOfProducts"]].max()
print(c1)


plt.figure(figsize=(10, 6))
age_gender_churn = df.groupby(["Age", "Gender"])["Exited"].mean().reset_index()

sns.lineplot(data=age_gender_churn, x="Age", y="Exited", hue="Gender", palette="Set1", marker="o")
plt.title("Churn Rate Trend by Age and Gender")
plt.xlabel("Age")
plt.ylabel("Average Churn Rate")
plt.tight_layout()
plt.show()


credit_card_counts = df["HasCrCard"].value_counts().sort_index()  # 0 = No, 1 = Yes

labels = ["No Credit Card", "Has Credit Card"]
colors = ["lightcoral", "lightgreen"]
sns.barplot(x=labels, y=credit_card_counts.values, palette=colors)
plt.title("Distribution of Credit Card Users")
plt.ylabel("Number of Customers")
plt.xlabel("Credit Card Ownership")
plt.show()
print()

#how many males and females using credit card and they are active member
filtred = df[(df["HasCrCard"] == 1) & (df["IsActiveMember"] == 1)]
fil = filtred[["Gender","HasCrCard","IsActiveMember"]].value_counts()
print("Using credit card and active: \n",fil)
print()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="CreditScore", y="EstimatedSalary", hue="Exited", palette="Set1")
plt.title("Credit Score vs Estimated Salary (Churn Highlighted)")
plt.xlabel("Credit Score")
plt.ylabel("Estimated Salary")
plt.legend(title="Exited")
plt.tight_layout()
plt.show()

gen = df.groupby("Gender")["HasCrCard"].sum()
print()
print(gen)

palette = ["pink","grey"]
sns.barplot(x = gen.index,y = gen.values,hue = gen.index,legend = False,palette=palette)
plt.xlabel("gender")
plt.ylabel("Has credit card")
plt.title("Number of credit card holders by Gender")
plt.show()

#checking the variations of the dat
rel = df.groupby("Gender")["EstimatedSalary"].std()
print()
print(rel)

#correlation btw age,creditscore,numofproducts
cor = df[["Age","CreditScore"]].corr()
print(cor)


#colors = ["pink","violet"]
#plt.bar(gen.index,gen.values,color = colors,edgecolor  = "black")
#plt.legend()
#plt.xlabel("Gender")
#plt.ylabel("Has Credit Card")
#plt.show()

sns.histplot(data = df,x = "Age",bins = 10,hue = "HasCrCard",kde = True,palette=["pink","blue"])
plt.xlabel("Age")
plt.title("Age Distribution of Customers by Credit Card Ownership")
plt.show()

#sns.stripplot(data = df,x = "Geography",y = "HasCrCard",hue = "Gender",dodge = True)
sns.catplot(data = df,x = "HasCrCard",hue = "Geography",kind="bar")
plt.xlabel("HasCreditCard")
plt.ylabel("count")
plt.title("Credit Card Ownership by Geography")
plt.show()

plt.figure(figsize=(10,5))
Gender = "Male"
df_year = df[df["Gender"] == Gender].nlargest(1,"CreditScore") #TOP 1 COUNTRIES
plt.bar(df_year["Geography"],df_year["Age"],color="pink",label=Gender)
plt.xlabel("Country")
plt.ylabel("Country (USD)")
plt.title(f"Top 1 countries by GDP in {Gender}")
plt.xticks(rotation=45)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Age", y="Balance", hue="Exited", palette="viridis", alpha=0.5)
plt.title("Age vs Balance Colored by Churn")
plt.xlabel("Customer Age")
plt.ylabel("Account Balance")
plt.legend(title="Exited")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix,cmap="viridis",annot=True,fmt=".2f")
plt.show()


sns.catplot(data=df, kind="bar",x="Geography", y="HasCrCard",hue="NumOfProducts", col="Geography", height=2)
plt.xlabel("Geography")
plt.ylabel("HasCreditCard")
plt.suptitle("Credit Card and Number of Products Ownership Across Regions")
plt.show()



#including only the selected columns
cols = ["Age","HasCrCard","NumOfProducts","IsActiveMember"]
sns.pairplot(data = df,hue = "Geography",vars=cols)
plt.show()



from statsmodels.stats.weightstats import ztest

# Split the data
churned = df[df["Exited"] == 1]["CreditScore"]
not_churned = df[df["Exited"] == 0]["CreditScore"]

# Perform z-test
z_score, p_value = ztest(churned, not_churned, alternative='two-sided')

print(f"Z-score: {z_score:.2f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in CreditScore between churned and non-churned customers.")
else:
    print("Fail to reject the null hypothesis: No significant difference in CreditScore between churned and non-churned customers.")


numeric_cols = df.select_dtypes(include=[np.number])

#compute skewness = numeric_cols.skew()
skewness = numeric_cols.skew()
print("\nSkewness of Numeric: \n",skewness)

#visuliztion to check skewness using histogram
#Plot histograms to visualize skewness

df.hist(bins = 20,color = 'darkviolet',grid=False,edgecolor = "black")
plt.title("Histogram for all numeric columns")
plt.xticks(rotation = 45)
plt.show()



#IQR is better choice for the datasets whenever there is skewness in the data(positive/negative)
#axis = 0 for cols adn axis = 1 for rows
#compute Q1(25th percentile) adn Q3 (75th percentile)

Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1
#define outliers as values outside 1.5 * IQR range
outliers = ((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR)))
outlier_counts = outliers.sum()

#print rows containing outliers
print("Outliers Detected:\n",df[outliers.any(axis=1)])

#count total outliers rows
total = outliers.any(axis=0).sum()

#print results
print("Outlier count per column:\n",outlier_counts)
print("\nTotal number of rows containing outliers:",total)

sns.boxplot(numeric_cols,palette="viridis")
plt.show()


