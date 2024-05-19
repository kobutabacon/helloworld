import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"
import streamlit as st

#read data
data = pd.read_csv('retail_price.csv')

#st.title
st.title('Retail Price Optimization')

#チェックボックスでdfの表示制御する。
data_frame = st.checkbox("show data_frame")
if data_frame:
    st.write(data)

#vis.histogram
fig = px.histogram(data, 
                x='total_price', 
                nbins=20, 
                title='Distribution of Total Price')
st.plotly_chart(fig)

#vis.Box Plot of Unit Price
fig_1 = px.box(data, 
                y='unit_price', 
                title='Box Plot of Unit Price')
st.plotly_chart(fig_1)

#vis.scatterplot of qty & total_price
fig_2 = px.scatter(data, 
                x='qty', 
                y='total_price', 
                title='Quantity vs Total Price', trendline="ols")
st.plotly_chart(fig_2)

#vis. Average Total Price by Product Category
fig_3 = px.bar(data, 
                x='product_category_name', 
                y='total_price', 
                title='Average Total Price by Product Category')
st.plotly_chart(fig_3)

#vis. Box Plot of Total Price by Weekday
fig_4 = px.box(data, 
                x='weekday', 
                y='total_price', 
                title='Box Plot of Total Price by Weekday')
st.plotly_chart(fig_4)

#vis. corr by heatmap
fig_5 = px.box(data, 
                x='holiday', 
                y='total_price', 
                title='Box Plot of Total Price by Holiday')
st.plotly_chart(fig_5)

#	vis. corr by heatmap
correlation_matrix = data.corr(numeric_only = True)
fig_6 = go.Figure(go.Heatmap(x=correlation_matrix.columns, 
                           y=correlation_matrix.columns, 
                           z=correlation_matrix.values))
fig_6.update_layout(title='Correlation Heatmap of Numerical Features')
st.plotly_chart(fig_6)

#	vis. Average Competitor Price Difference by Product Category
data['comp_price_diff'] = data['unit_price'] - data['comp_1'] 
avg_price_diff_by_category = data.groupby('product_category_name')['comp_price_diff'].mean().reset_index()
fig_7 = px.bar(avg_price_diff_by_category, 
                x='product_category_name', 
                y='comp_price_diff', 
                title='Average Competitor Price Difference by Product Category')
fig_7.update_layout(
                xaxis_title='Product Category',
                yaxis_title='Average Competitor Price Difference'
)
st.plotly_chart(fig_7)

#set decision tree model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
X = data[['qty', 'unit_price', 'comp_1', 
          'product_score', 'comp_price_diff']]
y = data['total_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=42)
# Train a linear regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#vis. scatter chart of correration
fig_8 = go.Figure()
fig_8.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', 
                            marker=dict(color='blue'), 
                            name='Predicted vs. Actual Retail Price'))
fig_8.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], 
                             mode='lines', 
                            marker=dict(color='red'), 
                            name='Ideal Prediction'))
fig_8.update_layout(
    title='Predicted vs. Actual Retail Price',
    xaxis_title='Actual Retail Price',
    yaxis_title='Predicted Retail Price'
)
st.plotly_chart(fig_8)
