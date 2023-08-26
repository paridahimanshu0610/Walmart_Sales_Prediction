#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


# Barplot to visualize the relation between 2 categorical variables
def plotCat2Cat(df, x_name, y_name):
    # Count the occurrences of each combination of categories
    count_df = df.groupby([y_name, x_name]).size().reset_index(name='Count')
    
    # Create a bar plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_name, y='Count', hue=y_name, data=count_df)
    
    plot_title = 'Count of ' + y_name + ' by ' + x_name
    plt.title(plot_title)
    plt.xlabel(x_name)
    plt.ylabel('Count')
    plt.legend(title=y_name)
    
    if df[x_name].nunique()>20:  # Rotating the x-labels by 90 deg if there more than 20 xlabels
        plt.xticks(rotation=90)
    
    plt.show()


# In[14]:


# Boxplot to visualize the relation between a categorical and a numerical variable
def plotCat2Num(df, x_name, y_name):
    # Create a boxplot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_name, y=y_name, data=df)
    
    plot_title = 'Boxplot of ' + y_name + ' by ' + x_name
    plt.title(plot_title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


# In[1]:


# Countplot for categorical features with percentage label on the bars
def count_plot(df, col_name):
    target_percentage = (df[col_name].value_counts()/df[col_name].count()) * 100

    # Create the count plot with percentages as labels
    plt.figure(figsize=[8, 4])
    
    count_order = df[col_name].value_counts().index
    ax = sns.countplot(data=df, x=col_name, order=count_order)
    plt.title(col_name + ' Distribution')
    plt.ylabel('Count')

    # Add percentage labels to the bars using target_percentage
    for p, percentage in zip(ax.patches, target_percentage):
        ax.annotate(f"{percentage:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.xticks(rotation=10)
    plt.show()

