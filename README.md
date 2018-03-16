
# Searching for Correlations in High School Graduations

Which factors have the most correlation on graduation from high school? The least? 
Are there any factors that increases the number of students that graduate high school (positive trend)?

We will be considering factors such as household income, race, teacher salary, school funding, pregnancy rate, and SAT scores.


```python
# dependencies
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
%matplotlib inline
```


```python
# plotly credentials

plotly.tools.set_credentials_file(username="abi.mvasquez", api_key="pX9WEUeT7jYv9HAtbhHX")
```

### School Funding (Revenue) by State ###


```python
# uploading csv and creating dataframe
rev_data = os.path.join('Resources','Stfis14_1a.csv')
rev_data_df = pd.read_csv(rev_data)
rev_data_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SURVYEAR</th>
      <th>FIPS</th>
      <th>STABR</th>
      <th>STNAME</th>
      <th>R1A</th>
      <th>R1B</th>
      <th>R1C</th>
      <th>R1D</th>
      <th>R1E</th>
      <th>R1F</th>
      <th>...</th>
      <th>A14B</th>
      <th>PPE15</th>
      <th>MEMBR13</th>
      <th>ARRASTE1</th>
      <th>ARRATE5</th>
      <th>ARRAE81Z</th>
      <th>ARRATE10</th>
      <th>ARRASTE6</th>
      <th>ARRATLEIZ</th>
      <th>ARRASTE4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>1</td>
      <td>AL</td>
      <td>Alabama</td>
      <td>-2</td>
      <td>-2</td>
      <td>1128860063</td>
      <td>621382410</td>
      <td>6131559</td>
      <td>1033208</td>
      <td>...</td>
      <td>706566</td>
      <td>8767</td>
      <td>746204</td>
      <td>805460</td>
      <td>1404270</td>
      <td>17206</td>
      <td>0</td>
      <td>15000</td>
      <td>107591</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>2</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>-2</td>
      <td>-2</td>
      <td>295232764</td>
      <td>161453827</td>
      <td>142222</td>
      <td>159692</td>
      <td>...</td>
      <td>-2</td>
      <td>19699</td>
      <td>130944</td>
      <td>20946</td>
      <td>46923</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>4</td>
      <td>AZ</td>
      <td>Arizona</td>
      <td>2979308590</td>
      <td>-2</td>
      <td>-2</td>
      <td>66772046</td>
      <td>5748335</td>
      <td>18493579</td>
      <td>...</td>
      <td>-2</td>
      <td>7783</td>
      <td>1102445</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014</td>
      <td>5</td>
      <td>AR</td>
      <td>Arkansas</td>
      <td>1619649401</td>
      <td>1223591</td>
      <td>-2</td>
      <td>11117198</td>
      <td>10086128</td>
      <td>2445184</td>
      <td>...</td>
      <td>-2</td>
      <td>9946</td>
      <td>489979</td>
      <td>1175969</td>
      <td>4031513</td>
      <td>0</td>
      <td>72883</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>6</td>
      <td>CA</td>
      <td>California</td>
      <td>18407152429</td>
      <td>933627693</td>
      <td>168400</td>
      <td>-2</td>
      <td>1117595</td>
      <td>371734184</td>
      <td>...</td>
      <td>-2</td>
      <td>9740</td>
      <td>6312623</td>
      <td>22075494</td>
      <td>37624403</td>
      <td>402380</td>
      <td>166270</td>
      <td>0</td>
      <td>29240684</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 159 columns</p>
</div>




```python
# filtering out the wanted data
rev_data_df = rev_data_df[["SURVYEAR", "STABR", "STNAME", "STR1", "R3", "STR4", "TR", "E11", "E11A"]]
rev_data_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SURVYEAR</th>
      <th>STABR</th>
      <th>STNAME</th>
      <th>STR1</th>
      <th>R3</th>
      <th>STR4</th>
      <th>TR</th>
      <th>E11</th>
      <th>E11A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>AL</td>
      <td>Alabama</td>
      <td>2479526880</td>
      <td>4065545836</td>
      <td>838649689</td>
      <td>7396933084</td>
      <td>2495972800</td>
      <td>1775677776</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>529595705</td>
      <td>1835601093</td>
      <td>312161832</td>
      <td>2677358630</td>
      <td>688001845</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>AZ</td>
      <td>Arizona</td>
      <td>3899570304</td>
      <td>4217359201</td>
      <td>1203567314</td>
      <td>9594427770</td>
      <td>3039539648</td>
      <td>2013800192</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014</td>
      <td>AR</td>
      <td>Arkansas</td>
      <td>1872187527</td>
      <td>2665329194</td>
      <td>592246357</td>
      <td>5133841370</td>
      <td>1818164198</td>
      <td>1170635549</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>CA</td>
      <td>California</td>
      <td>23107204742</td>
      <td>39293076276</td>
      <td>6942639777</td>
      <td>69342920795</td>
      <td>23880538763</td>
      <td>17076582010</td>
    </tr>
  </tbody>
</table>
</div>




```python
# using csv file key to rename columns
renamed_df = rev_data_df.rename(columns={"SURVYEAR":"YEAR", "STABR":"ST", "STNAME":"STATE", 
                                         "STR1":"LOCAL REVENUES SUBTOTAL", 
                                        "R3":"STATE REVENUES", "STR4":"FEDERAL REVENUES SUBTOTAL", 
                                        "TR":"TOTAL REVENUES FROM ALL SOURCES", 
                                         "E11":"INSTRUCTIONAL EXPENDITURES SALARIES", 
                                        "E11A":"TEACHER SALARIES REGULAR PROGRAMS"})
renamed_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>ST</th>
      <th>STATE</th>
      <th>LOCAL REVENUES SUBTOTAL</th>
      <th>STATE REVENUES</th>
      <th>FEDERAL REVENUES SUBTOTAL</th>
      <th>TOTAL REVENUES FROM ALL SOURCES</th>
      <th>INSTRUCTIONAL EXPENDITURES SALARIES</th>
      <th>TEACHER SALARIES REGULAR PROGRAMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>AL</td>
      <td>Alabama</td>
      <td>2479526880</td>
      <td>4065545836</td>
      <td>838649689</td>
      <td>7396933084</td>
      <td>2495972800</td>
      <td>1775677776</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>529595705</td>
      <td>1835601093</td>
      <td>312161832</td>
      <td>2677358630</td>
      <td>688001845</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>AZ</td>
      <td>Arizona</td>
      <td>3899570304</td>
      <td>4217359201</td>
      <td>1203567314</td>
      <td>9594427770</td>
      <td>3039539648</td>
      <td>2013800192</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014</td>
      <td>AR</td>
      <td>Arkansas</td>
      <td>1872187527</td>
      <td>2665329194</td>
      <td>592246357</td>
      <td>5133841370</td>
      <td>1818164198</td>
      <td>1170635549</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>CA</td>
      <td>California</td>
      <td>23107204742</td>
      <td>39293076276</td>
      <td>6942639777</td>
      <td>69342920795</td>
      <td>23880538763</td>
      <td>17076582010</td>
    </tr>
  </tbody>
</table>
</div>




```python
# school funding (revenue) by state data in map
trc = dict(type='choropleth', locations=renamed_df["ST"], locationmode='USA-states', colorscale='Greens',
          z=renamed_df["TOTAL REVENUES FROM ALL SOURCES"], colorbar = dict(
            title = "Billions of Dollars"))

layout=dict(geo=dict(scope='usa'),  title = 'School Funding (Revenue) by State')
map=go.Figure(data=[trc], layout=layout)

plt.savefig('SchoolFunding.png')
py.iplot(map)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~abi.mvasquez/282.embed" height="525px" width="100%"></iframe>




    <matplotlib.figure.Figure at 0x1138e65f8>


### Teen Pregnancy Rate by State ###


```python
# import super janky csv and create dataframe
#https://www.cdc.gov/nchs/pressroom/sosmap/teen-births/teenbirths.htm
# birth rate is (number of births) x 1000 / estimated population at mid-year
preg_data = os.path.join('Resources','TEENBIRTHS2016.csv')
preg_data_df = pd.read_csv(preg_data)
preg_data_df = preg_data_df.rename(columns={"STATE":"ST"})
preg_data_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>RATE</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>28.4</td>
      <td>/nchs/pressroom/states/alabama/alabama.htm</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>25.8</td>
      <td>/nchs/pressroom/states/alaska/alaska.htm</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>23.6</td>
      <td>/nchs/pressroom/states/arizona/arizona.htm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>34.6</td>
      <td>/nchs/pressroom/states/arkansas/arkansas.htm</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>17.0</td>
      <td>/nchs/pressroom/states/california/california.htm</td>
    </tr>
  </tbody>
</table>
</div>




```python
# teen pregnancy rate by state data map
trc = dict(type='choropleth', locations=preg_data_df["ST"], locationmode='USA-states', colorscale='Reds',
          z=preg_data_df["RATE"], colorbar = dict(
            title = "Percent (%)"))

layout=dict(geo=dict(scope='usa'),  title = 'Teen Pregnancy Rates per State')
map=go.Figure(data=[trc], layout=layout)

plt.savefig('PregnancyRate.png')
py.iplot(map)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~abi.mvasquez/284.embed" height="525px" width="100%"></iframe>




    <matplotlib.figure.Figure at 0x1139cd208>


### Spending per Student by State ###


```python
# import csv and create dataframe
#https://www.census.gov/data/tables/2014/econ/school-finances/secondary-education-finance.html
stu_spend = os.path.join('Resources','studentspending.csv')
stu_spend_df = pd.read_csv(stu_spend)
stu_spend_df = stu_spend_df.rename(columns={"State":"ST"})
stu_spend_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>Spending per Student</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DC</td>
      <td>29865.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NY</td>
      <td>23326.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CT</td>
      <td>20576.57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NJ</td>
      <td>20525.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AK</td>
      <td>20403.77</td>
    </tr>
  </tbody>
</table>
</div>




```python
# spending per student by state data map
trc = dict(type='choropleth', locations=stu_spend_df["ST"], locationmode='USA-states', colorscale='Greens',
          z=stu_spend_df["Spending per Student"], colorbar = dict(title = "Thousands of Dollars"))
lyt=dict(geo=dict(scope='usa'), title = 'Spending per Student by State')
map=go.Figure(data=[trc], layout=lyt)

plt.savefig('SpendingPerStudent.png')
py.iplot(map)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~abi.mvasquez/286.embed" height="525px" width="100%"></iframe>




    <matplotlib.figure.Figure at 0x1139178d0>


### Average SAT Scores by State ###


```python
# import csv and create dataframe
sat_scores = os.path.join('Resources', 'SATscores.csv')
sat_df = pd.read_csv(sat_scores)
sat_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>State1</th>
      <th>Average New SAT Score</th>
      <th>Participation Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>Alabama</td>
      <td>998</td>
      <td>7%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>Alaska</td>
      <td>1037</td>
      <td>54%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>Arizona</td>
      <td>1045</td>
      <td>36%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>Arkansas</td>
      <td>1034</td>
      <td>4%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>California</td>
      <td>1053</td>
      <td>60%</td>
    </tr>
  </tbody>
</table>
</div>




```python
satdf = sat_df.rename(columns={"State":"ST"})
satdf.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>State1</th>
      <th>Average New SAT Score</th>
      <th>Participation Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>Alabama</td>
      <td>998</td>
      <td>7%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>Alaska</td>
      <td>1037</td>
      <td>54%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>Arizona</td>
      <td>1045</td>
      <td>36%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>Arkansas</td>
      <td>1034</td>
      <td>4%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>California</td>
      <td>1053</td>
      <td>60%</td>
    </tr>
  </tbody>
</table>
</div>




```python
# average SAT scores by state data map
satdf = dict(type='choropleth', locations=satdf['ST'], locationmode='USA-states', colorscale='Blues',
        z=satdf['Average New SAT Score'], colorbar = dict(
          title = "Average SAT Score"))

layout=dict(geo=dict(scope='usa'),  title = 'Average SAT Score per State')
map=go.Figure(data=[satdf], layout=layout)

plt.savefig('SATScores.png')
py.iplot(map)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~abi.mvasquez/288.embed" height="525px" width="100%"></iframe>




    <matplotlib.figure.Figure at 0x1139ff9b0>


### Average Graduation Rates by State ###


```python
# import csv and create dataframe
grad_rates = os.path.join('Resources','gradrates.csv')
grad_df = pd.read_csv(grad_rates)
grad_df = grad_df.rename(columns={"State":"ST"})
grad_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>State1</th>
      <th>Average Rate</th>
      <th>American Indian/Alaska Native</th>
      <th>Asian/Pacific Islander</th>
      <th>Hispanic</th>
      <th>Black</th>
      <th>White</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>Alabama</td>
      <td>0.871</td>
      <td>0.900</td>
      <td>0.910</td>
      <td>0.870</td>
      <td>0.845</td>
      <td>0.886</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>Alaska</td>
      <td>0.761</td>
      <td>0.640</td>
      <td>0.810</td>
      <td>0.760</td>
      <td>0.740</td>
      <td>0.808</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>Arizona</td>
      <td>0.795</td>
      <td>0.677</td>
      <td>0.890</td>
      <td>0.764</td>
      <td>0.755</td>
      <td>0.840</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>Arkansas</td>
      <td>0.870</td>
      <td>0.870</td>
      <td>0.870</td>
      <td>0.857</td>
      <td>0.815</td>
      <td>0.892</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>California</td>
      <td>0.830</td>
      <td>0.740</td>
      <td>0.929</td>
      <td>0.800</td>
      <td>0.730</td>
      <td>0.880</td>
    </tr>
  </tbody>
</table>
</div>




```python
sorted_grad = grad_df.sort_values(by=["Average Rate"], ascending=False)
sorted_grad.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>State1</th>
      <th>Average Rate</th>
      <th>American Indian/Alaska Native</th>
      <th>Asian/Pacific Islander</th>
      <th>Hispanic</th>
      <th>Black</th>
      <th>White</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>KS</td>
      <td>Iowa</td>
      <td>0.913</td>
      <td>0.81</td>
      <td>0.910</td>
      <td>0.850</td>
      <td>0.800</td>
      <td>0.929</td>
    </tr>
    <tr>
      <th>30</th>
      <td>NM</td>
      <td>New Jersey</td>
      <td>0.901</td>
      <td>0.83</td>
      <td>0.967</td>
      <td>0.833</td>
      <td>0.821</td>
      <td>0.942</td>
    </tr>
    <tr>
      <th>48</th>
      <td>WI</td>
      <td>West Virginia</td>
      <td>0.898</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.890</td>
      <td>0.880</td>
      <td>0.899</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NV</td>
      <td>Nebraska</td>
      <td>0.893</td>
      <td>0.74</td>
      <td>0.810</td>
      <td>0.818</td>
      <td>0.790</td>
      <td>0.926</td>
    </tr>
    <tr>
      <th>43</th>
      <td>UT</td>
      <td>Texas</td>
      <td>0.891</td>
      <td>0.87</td>
      <td>0.954</td>
      <td>0.869</td>
      <td>0.854</td>
      <td>0.934</td>
    </tr>
  </tbody>
</table>
</div>



### Average Teacher Salary by State ###


```python
teacher_salary = os.path.join('Resources', 'teachersalary.csv')
teacher_df = pd.read_csv(teacher_salary)
teacher_df = teacher_df.rename(columns={"State":"ST"})
teacher_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>STATE1</th>
      <th>AVERAGE STARTING SALARY</th>
      <th>AVERAGE SALARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>Alabama</td>
      <td>36198</td>
      <td>47949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>Alaska</td>
      <td>44166</td>
      <td>65468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>Arkansas</td>
      <td>32691</td>
      <td>46632</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>Arizona</td>
      <td>31874</td>
      <td>49885</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>California</td>
      <td>41259</td>
      <td>69324</td>
    </tr>
  </tbody>
</table>
</div>



### High School Grad Rates ###


```python
# importing csv and creating dataframe ***need csv

csv = os.path.join('Resources', 'Hsgradrates2.csv')
hsgraddf = pd.read_csv(csv)
hsgraddf.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>States</th>
      <th>2014-15</th>
      <th>White</th>
      <th>Black</th>
      <th>Hispanic</th>
      <th>Asian Pacific Islander</th>
      <th>Asian Pacific Islander</th>
      <th>Students with disabil-ities</th>
      <th>Limited English Proficient</th>
      <th>Economically disadvan-taged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>United States</td>
      <td>83</td>
      <td>88</td>
      <td>75</td>
      <td>78</td>
      <td>90.0</td>
      <td>72.0</td>
      <td>65</td>
      <td>65</td>
      <td>76</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Alabama</td>
      <td>89</td>
      <td>91</td>
      <td>87</td>
      <td>90</td>
      <td>93.0</td>
      <td>90.0</td>
      <td>72</td>
      <td>75</td>
      <td>85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Alaska</td>
      <td>76</td>
      <td>80</td>
      <td>71</td>
      <td>72</td>
      <td>83.0</td>
      <td>64.0</td>
      <td>57</td>
      <td>56</td>
      <td>67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Arizona</td>
      <td>77</td>
      <td>83</td>
      <td>73</td>
      <td>73</td>
      <td>87.0</td>
      <td>67.0</td>
      <td>64</td>
      <td>34</td>
      <td>73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Arkansas</td>
      <td>85</td>
      <td>87</td>
      <td>78</td>
      <td>85</td>
      <td>86.0</td>
      <td>80.0</td>
      <td>82</td>
      <td>86</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
hsgraddf["ST"] = ['US','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
                  'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM',
                  'NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA',
                  'WV','WI','WY']
hsgraddf = hsgraddf[["ST","States", "2014-15","White", "Black", "Hispanic", "Asian Pacific Islander", 
                     "Asian Pacific Islander", "Students with disabil-ities", "Limited English Proficient",
                    "Economically disadvan-taged"]]
hsgraddf.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>States</th>
      <th>2014-15</th>
      <th>White</th>
      <th>Black</th>
      <th>Hispanic</th>
      <th>Asian Pacific Islander</th>
      <th>Asian Pacific Islander</th>
      <th>Students with disabil-ities</th>
      <th>Limited English Proficient</th>
      <th>Economically disadvan-taged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>United States</td>
      <td>83</td>
      <td>88</td>
      <td>75</td>
      <td>78</td>
      <td>72.0</td>
      <td>72.0</td>
      <td>65</td>
      <td>65</td>
      <td>76</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>Alabama</td>
      <td>89</td>
      <td>91</td>
      <td>87</td>
      <td>90</td>
      <td>90.0</td>
      <td>90.0</td>
      <td>72</td>
      <td>75</td>
      <td>85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AK</td>
      <td>Alaska</td>
      <td>76</td>
      <td>80</td>
      <td>71</td>
      <td>72</td>
      <td>64.0</td>
      <td>64.0</td>
      <td>57</td>
      <td>56</td>
      <td>67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AZ</td>
      <td>Arizona</td>
      <td>77</td>
      <td>83</td>
      <td>73</td>
      <td>73</td>
      <td>67.0</td>
      <td>67.0</td>
      <td>64</td>
      <td>34</td>
      <td>73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AR</td>
      <td>Arkansas</td>
      <td>85</td>
      <td>87</td>
      <td>78</td>
      <td>85</td>
      <td>80.0</td>
      <td>80.0</td>
      <td>82</td>
      <td>86</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
trc = dict(type='choropleth', locations=['US','AL','AK','AZ','AR','CA','CO','CT','DE','DOC','FL','GA','HI','ID','IL','IN','IA',
                                         'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM',
                                         'NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA',
                                         'WV','WI','WY']
, locationmode='USA-states', colorscale='YlOrRd',
           z=hsgraddf['2014-15'], colorbar = dict(
            title = 'Percent (%)'))
lyt=dict(geo=dict(scope='usa'))
map=go.Figure(data=[trc], layout=lyt)

py.iplot(map)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~abi.mvasquez/290.embed" height="525px" width="100%"></iframe>



### Graduation by Income ###

**Adjusted Cohort Graduation Rate(ACGR)**

**"http://gradnation.americaspromise.org/sites/default/files/d8/2017-05/Appendix_G.pdf", As part of an audit conducted by education officials from Alabama in December 2016, it was announced that the graduation rates of the state were inaccurate because the reports had been calculated improperly, hence the rates of this report may have some inflation, since the graduation rates of Alabama is still included here.**

**2015(%) = number of low-income students, which was divided by the total cohort size for each state. Estimated non low-income ACGR 2015 (%) = the estimated graduates from all students minus low-income graduates 2015 (%) = the number of low-income students divided by the total cohort size within each state. Estimated Non-Low-Income ACGR (%) = the estimated graduates from all students minus low-income graduates divided by the estimated total cohort of all students minus low-income within the cohort (i.e., using state level ACGRs). Gap Change Between Non-Low-Income and Low-Income ACGR (Percentage Points), 2011-15 = the gap between the estimated non-low-income and low-income ACGRs from 2010-11 to 2013-15. Therefore, positive values indicate gap closure and negative values indicate gap widening. Sources: U.S. Department of Education through provisional data file of SY2010-11 and SY 2014-15 State Level Four-Year Regulatory Adjusted Cohort Graduation Rates and Cohort Counts. Retrieved on November 6, 2016 from http://eddataexpress.ed.gov/state-tables-main.cfm".**


```python
#Data Source
# http://gradnation.americaspromise.org/sites/default/files/d8/2017-05/Appendix_G.pdf
# used adobe acrobat pro to convert the PDF to excel format and then saved excel as csv
csv_glor = os.path.join('Resources', 'Main_data_AppendixG.csv')
gh = pd.read_csv(csv_glor)
gh.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Abb</th>
      <th>State</th>
      <th>Gap between Non-Low Income 
and Low-Income ACGR (Percentage Points), 2011</th>
      <th>Overall 2015 
ACGR(%)</th>
      <th>Percent of Low-Income 
Students in the 
Cohort, 2015 (%)</th>
      <th>Estimated 
Non-Low-Income 2015 ACGR(%)</th>
      <th>Low-Income 2015
ACGR(%)</th>
      <th>Gap between Non-Low Income 
and Low-Income ACGR
(Percentage Points, 2015</th>
      <th>Gap Change between Non-Low-Income
 and Low-Income ACGR
(Percentage points), 2011-15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>Alabama*</td>
      <td>19.73</td>
      <td>89.3%</td>
      <td>49.5%</td>
      <td>93.8%</td>
      <td>84.7%</td>
      <td>9.1</td>
      <td>10.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>Alaska</td>
      <td>18.28</td>
      <td>75.6%</td>
      <td>35.8%</td>
      <td>80.6%</td>
      <td>66.6%</td>
      <td>14.0</td>
      <td>4.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>Arizona</td>
      <td>7.94</td>
      <td>77.4%</td>
      <td>39.7%</td>
      <td>80.2%</td>
      <td>73.1%</td>
      <td>7.1</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>Arkansas</td>
      <td>12.14</td>
      <td>84.9%</td>
      <td>49.6%</td>
      <td>88.1%</td>
      <td>81.7%</td>
      <td>6.4</td>
      <td>5.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>California</td>
      <td>15.49</td>
      <td>82.0%</td>
      <td>67.2%</td>
      <td>90.2%</td>
      <td>78.0%</td>
      <td>12.2</td>
      <td>3.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# printed this to show that the last row has no data
gh.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Abb</th>
      <th>State</th>
      <th>Gap between Non-Low Income 
and Low-Income ACGR (Percentage Points), 2011</th>
      <th>Overall 2015 
ACGR(%)</th>
      <th>Percent of Low-Income 
Students in the 
Cohort, 2015 (%)</th>
      <th>Estimated 
Non-Low-Income 2015 ACGR(%)</th>
      <th>Low-Income 2015
ACGR(%)</th>
      <th>Gap between Non-Low Income 
and Low-Income ACGR
(Percentage Points, 2015</th>
      <th>Gap Change between Non-Low-Income
 and Low-Income ACGR
(Percentage points), 2011-15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>WA</td>
      <td>Washington</td>
      <td>17.38</td>
      <td>78.2%</td>
      <td>51.2%</td>
      <td>88.8%</td>
      <td>68.1%</td>
      <td>20.7</td>
      <td>-3.3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>WV</td>
      <td>West Virginia</td>
      <td>19.86</td>
      <td>86.5%</td>
      <td>66.4%</td>
      <td>93.6%</td>
      <td>82.9%</td>
      <td>10.7</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>48</th>
      <td>WI</td>
      <td>Wisconsin</td>
      <td>18.00</td>
      <td>88.4%</td>
      <td>32.2%</td>
      <td>93.7%</td>
      <td>77.3%</td>
      <td>16.4</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>49</th>
      <td>WY</td>
      <td>Wyoming</td>
      <td>21.66</td>
      <td>79.3%</td>
      <td>39.6%</td>
      <td>88.0%</td>
      <td>66.0%</td>
      <td>22.0</td>
      <td>-0.4</td>
    </tr>
    <tr>
      <th>50</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# printed this to see accuracute columns spelling and hidden characters
gh.columns
```




    Index(['Abb', 'State',
           'Gap between Non-Low Income \nand Low-Income ACGR (Percentage Points), 2011',
           'Overall 2015 \nACGR(%)',
           'Percent of Low-Income \nStudents in the \nCohort, 2015 (%)',
           'Estimated \nNon-Low-Income 2015 ACGR(%)', 'Low-Income 2015\nACGR(%)',
           'Gap between Non-Low Income \nand Low-Income ACGR\n(Percentage Points, 2015',
           'Gap Change between Non-Low-Income\n and Low-Income ACGR\n(Percentage points), 2011-15'],
          dtype='object')




```python
#Cleanup

#removed percentage signs, removed † and replace with NaN, format cells as float
gh['Low-Income 2015\nACGR(%)'] = gh['Low-Income 2015\nACGR(%)'].str.replace('%','').astype(float)
gh['Estimated \nNon-Low-Income 2015 ACGR(%)'] = gh['Estimated \nNon-Low-Income 2015 ACGR(%)'].str.replace('%','').astype(float)
gh['Gap Change between Non-Low-Income\n and Low-Income ACGR\n(Percentage points), 2011-15'] = gh['Gap Change between Non-Low-Income\n and Low-Income ACGR\n(Percentage points), 2011-15'].str.replace('†','NaN').astype(float)
gh['Overall 2015 \nACGR(%)'] = gh['Overall 2015 \nACGR(%)'].str.replace('%','').astype(float)

# drop the last column showing NaN since its not a state
gh = gh.drop(gh.index[len(gh)-1])
gh.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Abb</th>
      <th>State</th>
      <th>Gap between Non-Low Income 
and Low-Income ACGR (Percentage Points), 2011</th>
      <th>Overall 2015 
ACGR(%)</th>
      <th>Percent of Low-Income 
Students in the 
Cohort, 2015 (%)</th>
      <th>Estimated 
Non-Low-Income 2015 ACGR(%)</th>
      <th>Low-Income 2015
ACGR(%)</th>
      <th>Gap between Non-Low Income 
and Low-Income ACGR
(Percentage Points, 2015</th>
      <th>Gap Change between Non-Low-Income
 and Low-Income ACGR
(Percentage points), 2011-15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>VA</td>
      <td>Virginia</td>
      <td>17.06</td>
      <td>85.7</td>
      <td>31.8%</td>
      <td>90.5</td>
      <td>75.4</td>
      <td>15.1</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>46</th>
      <td>WA</td>
      <td>Washington</td>
      <td>17.38</td>
      <td>78.2</td>
      <td>51.2%</td>
      <td>88.8</td>
      <td>68.1</td>
      <td>20.7</td>
      <td>-3.3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>WV</td>
      <td>West Virginia</td>
      <td>19.86</td>
      <td>86.5</td>
      <td>66.4%</td>
      <td>93.6</td>
      <td>82.9</td>
      <td>10.7</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>48</th>
      <td>WI</td>
      <td>Wisconsin</td>
      <td>18.00</td>
      <td>88.4</td>
      <td>32.2%</td>
      <td>93.7</td>
      <td>77.3</td>
      <td>16.4</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>49</th>
      <td>WY</td>
      <td>Wyoming</td>
      <td>21.66</td>
      <td>79.3</td>
      <td>39.6%</td>
      <td>88.0</td>
      <td>66.0</td>
      <td>22.0</td>
      <td>-0.4</td>
    </tr>
  </tbody>
</table>
</div>



** Which 5 states are low household income high schoolers most likely to graduate? ** 
**Ans: Most likely; Texas, Kentucky, Iowa, Alabama, Indiana**

**Which 5 states are low household income high schoolers less likely to graduate
Less likely; Oregon, Wyoming, Colorado, Nevada, New Mexico***


```python
gh[ ['State','Low-Income 2015\nACGR(%)'] ].sort_values(by='Low-Income 2015\nACGR(%)', ascending=False).head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Low-Income 2015
ACGR(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>Texas</td>
      <td>85.6</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kentucky</td>
      <td>84.8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Iowa</td>
      <td>84.8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama*</td>
      <td>84.7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Indiana</td>
      <td>84.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
gh[ ['State','Low-Income 2015\nACGR(%)'] ].sort_values(by='Low-Income 2015\nACGR(%)', ascending=False).tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Low-Income 2015
ACGR(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>Oregon</td>
      <td>66.4</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Wyoming</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>65.5</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nevada</td>
      <td>63.7</td>
    </tr>
    <tr>
      <th>30</th>
      <td>New Mexico</td>
      <td>63.5</td>
    </tr>
  </tbody>
</table>
</div>



**What impact does income have on high school graduation rates?**

**Ans: The higher the household income, the greater the likelihood of graduation from high school**


```python
#gh.plot(x='Percent of Low-Income \nStudents in the \nCohort, 2015 (%)', y='Estimated \nNon-Low-Income 2015 ACGR(%)')
gh.plot(x="Abb",
        y=["Low-Income 2015\nACGR(%)",
          "Estimated \nNon-Low-Income 2015 ACGR(%)"], kind="bar", figsize=(20,20), fontsize=12, legend = True)

plt.title("Estimated High School graduation based on Household Income 2015", size = 22)
plt.xlabel("US States", size = 22)
plt.ylabel("Estimated Graduation Rates", size = 22)
plt.legend(loc=4, prop={'size': 55})
```




    <matplotlib.legend.Legend at 0x111ae65f8>




![png](output_38_1.png)


**What States have closed the gap between low-income and non-low-income graduation rates?**

**Ans: Alabama, West Virginia, Connecticut, Indiana, Georgia**


```python
gh[ ['State', 'Gap Change between Non-Low-Income\n and Low-Income ACGR\n(Percentage points), 2011-15'] ].sort_values(by='Gap Change between Non-Low-Income\n and Low-Income ACGR\n(Percentage points), 2011-15', ascending=False).head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Gap Change between Non-Low-Income
 and Low-Income ACGR
(Percentage points), 2011-15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama*</td>
      <td>10.6</td>
    </tr>
    <tr>
      <th>47</th>
      <td>West Virginia</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>7.9</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Indiana</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Georgia</td>
      <td>5.8</td>
    </tr>
  </tbody>
</table>
</div>



**positive values indicate gap closure and negative values indicate gap widening.**


```python
gh.plot(x="Abb", y='Gap Change between Non-Low-Income\n and Low-Income ACGR\n(Percentage points), 2011-15', kind="bar", figsize=(20,20), fontsize=12)

plt.title('Graduation gap between low-income and non-low-income 2011-2015', size = 24)
plt.xlabel('US Sates', size=24)
plt.ylabel('Adjusted Cohort Graduation Rate',size=24)
plt.legend(loc=0, prop={'size': 40})

```




    <matplotlib.legend.Legend at 0x1a1b3959e8>




![png](output_42_1.png)


### Merging Dataframes ###


```python
# merging school funding and pregnancy rate
merge_table = pd.merge(renamed_df, preg_data_df, on="ST", how="outer")
merge_table.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>ST</th>
      <th>STATE</th>
      <th>LOCAL REVENUES SUBTOTAL</th>
      <th>STATE REVENUES</th>
      <th>FEDERAL REVENUES SUBTOTAL</th>
      <th>TOTAL REVENUES FROM ALL SOURCES</th>
      <th>INSTRUCTIONAL EXPENDITURES SALARIES</th>
      <th>TEACHER SALARIES REGULAR PROGRAMS</th>
      <th>RATE</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>AL</td>
      <td>Alabama</td>
      <td>2479526880</td>
      <td>4065545836</td>
      <td>838649689</td>
      <td>7396933084</td>
      <td>2495972800</td>
      <td>1775677776</td>
      <td>28.4</td>
      <td>/nchs/pressroom/states/alabama/alabama.htm</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>AK</td>
      <td>Alaska</td>
      <td>529595705</td>
      <td>1835601093</td>
      <td>312161832</td>
      <td>2677358630</td>
      <td>688001845</td>
      <td>-1</td>
      <td>25.8</td>
      <td>/nchs/pressroom/states/alaska/alaska.htm</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>AZ</td>
      <td>Arizona</td>
      <td>3899570304</td>
      <td>4217359201</td>
      <td>1203567314</td>
      <td>9594427770</td>
      <td>3039539648</td>
      <td>2013800192</td>
      <td>23.6</td>
      <td>/nchs/pressroom/states/arizona/arizona.htm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014</td>
      <td>AR</td>
      <td>Arkansas</td>
      <td>1872187527</td>
      <td>2665329194</td>
      <td>592246357</td>
      <td>5133841370</td>
      <td>1818164198</td>
      <td>1170635549</td>
      <td>34.6</td>
      <td>/nchs/pressroom/states/arkansas/arkansas.htm</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>CA</td>
      <td>California</td>
      <td>23107204742</td>
      <td>39293076276</td>
      <td>6942639777</td>
      <td>69342920795</td>
      <td>23880538763</td>
      <td>17076582010</td>
      <td>17.0</td>
      <td>/nchs/pressroom/states/california/california.htm</td>
    </tr>
  </tbody>
</table>
</div>




```python
merge_table = merge_table[["ST","STATE REVENUES", 
                           "TOTAL REVENUES FROM ALL SOURCES","RATE"]]
```


```python
final_merge = merge_table.rename(columns={"RATE":"PREGNANCY RATE"})
final_merge.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>STATE REVENUES</th>
      <th>TOTAL REVENUES FROM ALL SOURCES</th>
      <th>PREGNANCY RATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>4065545836</td>
      <td>7396933084</td>
      <td>28.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>1835601093</td>
      <td>2677358630</td>
      <td>25.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>4217359201</td>
      <td>9594427770</td>
      <td>23.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>2665329194</td>
      <td>5133841370</td>
      <td>34.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>39293076276</td>
      <td>69342920795</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merging spending per student and average SAT
merge2 = pd.merge(satdf, stu_spend_df, on="ST", how="outer")
merge2.head()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>State1</th>
      <th>Average New SAT Score</th>
      <th>Participation Rate</th>
      <th>Spending per Student</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>Alabama</td>
      <td>998.0</td>
      <td>7%</td>
      <td>9938.84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>Alaska</td>
      <td>1037.0</td>
      <td>54%</td>
      <td>20403.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>Arizona</td>
      <td>1045.0</td>
      <td>36%</td>
      <td>8786.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>Arkansas</td>
      <td>1034.0</td>
      <td>4%</td>
      <td>10785.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>California</td>
      <td>1053.0</td>
      <td>60%</td>
      <td>11222.72</td>
    </tr>
  </tbody>
</table>
</div>




```python
merge2 = merge2[["ST","Average New SAT Score", "Spending per Student"]]
merge2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>Average New SAT Score</th>
      <th>Spending per Student</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>998.0</td>
      <td>9938.84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>1037.0</td>
      <td>20403.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>1045.0</td>
      <td>8786.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>1034.0</td>
      <td>10785.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>1053.0</td>
      <td>11222.72</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merging school funding, pregnancy rate, spending per student, and average SAT
merge3 = pd.merge(merge2, final_merge, on="ST", how="outer")
```


```python
# merging average graduation rates and average teacher salary
merge4 = pd.merge(sorted_grad, teacher_df, on="ST", how="outer")
merge4 = merge4[["ST","Average Rate", "American Indian/Alaska Native", "Asian/Pacific Islander", "Hispanic", "Black",
                "White", "AVERAGE STARTING SALARY", "AVERAGE SALARY"]]

merge4.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>Average Rate</th>
      <th>American Indian/Alaska Native</th>
      <th>Asian/Pacific Islander</th>
      <th>Hispanic</th>
      <th>Black</th>
      <th>White</th>
      <th>AVERAGE STARTING SALARY</th>
      <th>AVERAGE SALARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>0.913</td>
      <td>0.81</td>
      <td>0.910</td>
      <td>0.850</td>
      <td>0.800</td>
      <td>0.929</td>
      <td>34696</td>
      <td>51456</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NM</td>
      <td>0.901</td>
      <td>0.83</td>
      <td>0.967</td>
      <td>0.833</td>
      <td>0.821</td>
      <td>0.942</td>
      <td>34280</td>
      <td>55599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WI</td>
      <td>0.898</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.890</td>
      <td>0.880</td>
      <td>0.899</td>
      <td>33546</td>
      <td>55171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NV</td>
      <td>0.893</td>
      <td>0.74</td>
      <td>0.810</td>
      <td>0.818</td>
      <td>0.790</td>
      <td>0.926</td>
      <td>30778</td>
      <td>45947</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UT</td>
      <td>0.891</td>
      <td>0.87</td>
      <td>0.954</td>
      <td>0.869</td>
      <td>0.854</td>
      <td>0.934</td>
      <td>38091</td>
      <td>48110</td>
    </tr>
  </tbody>
</table>
</div>




```python
merge4_fin = merge4.rename(columns={"Average Rate":"Average Graduation Rate", 
                                        "American Indian/Alaska Native":"Average Graduation Rate (American Indian/Alaska Native)",
                                       "Asian/Pacific Islander":"Average Graduation Rate (Asian/Pacific Islander)",
                                       "Hispanic":"Average Graduation Rate (Hispanic)",
                                       "Black":"Average Graduation Rate (Black)",
                                       "White":"Average Graduation Rate (White)",
                                       "AVERAGE STARTING SALARY":"Average Starting Teacher Salary",
                                   "AVERAGE SALARY":"Average Teacher Salary"})
merge4_fin.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>Average Graduation Rate</th>
      <th>Average Graduation Rate (American Indian/Alaska Native)</th>
      <th>Average Graduation Rate (Asian/Pacific Islander)</th>
      <th>Average Graduation Rate (Hispanic)</th>
      <th>Average Graduation Rate (Black)</th>
      <th>Average Graduation Rate (White)</th>
      <th>Average Starting Teacher Salary</th>
      <th>Average Teacher Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>0.913</td>
      <td>0.81</td>
      <td>0.910</td>
      <td>0.850</td>
      <td>0.800</td>
      <td>0.929</td>
      <td>34696</td>
      <td>51456</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NM</td>
      <td>0.901</td>
      <td>0.83</td>
      <td>0.967</td>
      <td>0.833</td>
      <td>0.821</td>
      <td>0.942</td>
      <td>34280</td>
      <td>55599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WI</td>
      <td>0.898</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.890</td>
      <td>0.880</td>
      <td>0.899</td>
      <td>33546</td>
      <td>55171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NV</td>
      <td>0.893</td>
      <td>0.74</td>
      <td>0.810</td>
      <td>0.818</td>
      <td>0.790</td>
      <td>0.926</td>
      <td>30778</td>
      <td>45947</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UT</td>
      <td>0.891</td>
      <td>0.87</td>
      <td>0.954</td>
      <td>0.869</td>
      <td>0.854</td>
      <td>0.934</td>
      <td>38091</td>
      <td>48110</td>
    </tr>
  </tbody>
</table>
</div>




```python
compiled1 = pd.merge(merge4_fin, merge3, on="ST", how="outer")
```


```python
compiled1 = compiled1.rename(columns={"STATE REVENUES":"State Revenues ($)", 
                                        "TOTAL REVENUES FROM ALL SOURCES":"Total Revenues From All Sources ($)",
                                       "PREGNANCY RATE":"Pregnancy Rate (%)"})
compiled1.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>Average Graduation Rate</th>
      <th>Average Graduation Rate (American Indian/Alaska Native)</th>
      <th>Average Graduation Rate (Asian/Pacific Islander)</th>
      <th>Average Graduation Rate (Hispanic)</th>
      <th>Average Graduation Rate (Black)</th>
      <th>Average Graduation Rate (White)</th>
      <th>Average Starting Teacher Salary</th>
      <th>Average Teacher Salary</th>
      <th>Average New SAT Score</th>
      <th>Spending per Student</th>
      <th>State Revenues ($)</th>
      <th>Total Revenues From All Sources ($)</th>
      <th>Pregnancy Rate (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>0.913</td>
      <td>0.81</td>
      <td>0.910</td>
      <td>0.850</td>
      <td>0.800</td>
      <td>0.929</td>
      <td>34696.0</td>
      <td>51456.0</td>
      <td>1098.0</td>
      <td>11717.66</td>
      <td>3.298508e+09</td>
      <td>6.065210e+09</td>
      <td>21.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NM</td>
      <td>0.901</td>
      <td>0.83</td>
      <td>0.967</td>
      <td>0.833</td>
      <td>0.821</td>
      <td>0.942</td>
      <td>34280.0</td>
      <td>55599.0</td>
      <td>1104.0</td>
      <td>11025.66</td>
      <td>2.645457e+09</td>
      <td>3.779535e+09</td>
      <td>29.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WI</td>
      <td>0.898</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.890</td>
      <td>0.880</td>
      <td>0.899</td>
      <td>33546.0</td>
      <td>55171.0</td>
      <td>963.0</td>
      <td>12716.48</td>
      <td>4.981241e+09</td>
      <td>1.098072e+10</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NV</td>
      <td>0.893</td>
      <td>0.74</td>
      <td>0.810</td>
      <td>0.818</td>
      <td>0.790</td>
      <td>0.926</td>
      <td>30778.0</td>
      <td>45947.0</td>
      <td>1070.0</td>
      <td>9641.57</td>
      <td>1.560330e+09</td>
      <td>4.341723e+09</td>
      <td>24.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UT</td>
      <td>0.891</td>
      <td>0.87</td>
      <td>0.954</td>
      <td>0.869</td>
      <td>0.854</td>
      <td>0.934</td>
      <td>38091.0</td>
      <td>48110.0</td>
      <td>1026.0</td>
      <td>7714.19</td>
      <td>2.673267e+09</td>
      <td>4.905540e+09</td>
      <td>15.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_dataframe = compiled1.drop([38, 51, 52, 53, 54, 55, 56])
edu_fin = final_dataframe.reset_index()
```


```python
education_final = edu_fin.drop('index', axis=1)
education_final
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>Average Graduation Rate</th>
      <th>Average Graduation Rate (American Indian/Alaska Native)</th>
      <th>Average Graduation Rate (Asian/Pacific Islander)</th>
      <th>Average Graduation Rate (Hispanic)</th>
      <th>Average Graduation Rate (Black)</th>
      <th>Average Graduation Rate (White)</th>
      <th>Average Starting Teacher Salary</th>
      <th>Average Teacher Salary</th>
      <th>Average New SAT Score</th>
      <th>Spending per Student</th>
      <th>State Revenues ($)</th>
      <th>Total Revenues From All Sources ($)</th>
      <th>Pregnancy Rate (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>0.913</td>
      <td>0.810</td>
      <td>0.910</td>
      <td>0.850</td>
      <td>0.800</td>
      <td>0.929</td>
      <td>34696.0</td>
      <td>51456.0</td>
      <td>1098.0</td>
      <td>11717.66</td>
      <td>3.298508e+09</td>
      <td>6.065210e+09</td>
      <td>21.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NM</td>
      <td>0.901</td>
      <td>0.830</td>
      <td>0.967</td>
      <td>0.833</td>
      <td>0.821</td>
      <td>0.942</td>
      <td>34280.0</td>
      <td>55599.0</td>
      <td>1104.0</td>
      <td>11025.66</td>
      <td>2.645457e+09</td>
      <td>3.779535e+09</td>
      <td>29.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WI</td>
      <td>0.898</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0.890</td>
      <td>0.880</td>
      <td>0.899</td>
      <td>33546.0</td>
      <td>55171.0</td>
      <td>963.0</td>
      <td>12716.48</td>
      <td>4.981241e+09</td>
      <td>1.098072e+10</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NV</td>
      <td>0.893</td>
      <td>0.740</td>
      <td>0.810</td>
      <td>0.818</td>
      <td>0.790</td>
      <td>0.926</td>
      <td>30778.0</td>
      <td>45947.0</td>
      <td>1070.0</td>
      <td>9641.57</td>
      <td>1.560330e+09</td>
      <td>4.341723e+09</td>
      <td>24.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UT</td>
      <td>0.891</td>
      <td>0.870</td>
      <td>0.954</td>
      <td>0.869</td>
      <td>0.854</td>
      <td>0.934</td>
      <td>38091.0</td>
      <td>48110.0</td>
      <td>1026.0</td>
      <td>7714.19</td>
      <td>2.673267e+09</td>
      <td>4.905540e+09</td>
      <td>15.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MT</td>
      <td>0.890</td>
      <td>0.860</td>
      <td>0.920</td>
      <td>0.831</td>
      <td>0.790</td>
      <td>0.916</td>
      <td>31184.0</td>
      <td>41994.0</td>
      <td>1089.0</td>
      <td>11889.51</td>
      <td>8.325352e+08</td>
      <td>1.723235e+09</td>
      <td>23.7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LA</td>
      <td>0.886</td>
      <td>0.830</td>
      <td>0.930</td>
      <td>0.820</td>
      <td>0.809</td>
      <td>0.900</td>
      <td>35166.0</td>
      <td>50326.0</td>
      <td>1064.0</td>
      <td>12507.53</td>
      <td>3.794407e+09</td>
      <td>8.733819e+09</td>
      <td>30.6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TX</td>
      <td>0.885</td>
      <td>0.860</td>
      <td>0.930</td>
      <td>0.837</td>
      <td>0.823</td>
      <td>0.913</td>
      <td>34098.0</td>
      <td>48289.0</td>
      <td>1054.0</td>
      <td>10629.22</td>
      <td>2.212761e+10</td>
      <td>5.337715e+10</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WY</td>
      <td>0.882</td>
      <td>0.780</td>
      <td>0.890</td>
      <td>0.799</td>
      <td>0.642</td>
      <td>0.927</td>
      <td>32533.0</td>
      <td>46405.0</td>
      <td>1096.0</td>
      <td>19098.34</td>
      <td>9.651595e+08</td>
      <td>1.771864e+09</td>
      <td>26.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NJ</td>
      <td>0.882</td>
      <td>0.740</td>
      <td>0.920</td>
      <td>0.760</td>
      <td>0.780</td>
      <td>0.892</td>
      <td>30844.0</td>
      <td>48931.0</td>
      <td>1101.0</td>
      <td>20525.21</td>
      <td>1.112216e+10</td>
      <td>2.736382e+10</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>VA</td>
      <td>0.877</td>
      <td>0.000</td>
      <td>0.830</td>
      <td>0.890</td>
      <td>0.710</td>
      <td>0.884</td>
      <td>37848.0</td>
      <td>49869.0</td>
      <td>1093.0</td>
      <td>11846.67</td>
      <td>5.984788e+09</td>
      <td>1.504948e+10</td>
      <td>15.5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MA</td>
      <td>0.876</td>
      <td>0.820</td>
      <td>0.951</td>
      <td>0.765</td>
      <td>0.841</td>
      <td>0.924</td>
      <td>43235.0</td>
      <td>65265.0</td>
      <td>1057.0</td>
      <td>17896.06</td>
      <td>6.597170e+09</td>
      <td>1.681241e+10</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>OH</td>
      <td>0.875</td>
      <td>0.660</td>
      <td>0.880</td>
      <td>0.770</td>
      <td>0.770</td>
      <td>0.908</td>
      <td>43839.0</td>
      <td>75279.0</td>
      <td>1099.0</td>
      <td>14000.97</td>
      <td>1.040676e+10</td>
      <td>2.349424e+10</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MI</td>
      <td>0.875</td>
      <td>0.850</td>
      <td>0.927</td>
      <td>0.727</td>
      <td>0.789</td>
      <td>0.919</td>
      <td>31835.0</td>
      <td>48119.0</td>
      <td>1130.0</td>
      <td>12855.62</td>
      <td>1.121164e+10</td>
      <td>1.888372e+10</td>
      <td>17.7</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CT</td>
      <td>0.874</td>
      <td>0.890</td>
      <td>0.940</td>
      <td>0.764</td>
      <td>0.788</td>
      <td>0.925</td>
      <td>42924.0</td>
      <td>69766.0</td>
      <td>1126.0</td>
      <td>20576.57</td>
      <td>4.418595e+09</td>
      <td>1.101769e+10</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>AL</td>
      <td>0.871</td>
      <td>0.900</td>
      <td>0.910</td>
      <td>0.870</td>
      <td>0.845</td>
      <td>0.886</td>
      <td>36198.0</td>
      <td>47949.0</td>
      <td>998.0</td>
      <td>9938.84</td>
      <td>4.065546e+09</td>
      <td>7.396933e+09</td>
      <td>28.4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>AR</td>
      <td>0.870</td>
      <td>0.870</td>
      <td>0.870</td>
      <td>0.857</td>
      <td>0.815</td>
      <td>0.892</td>
      <td>31874.0</td>
      <td>49885.0</td>
      <td>1034.0</td>
      <td>10785.03</td>
      <td>2.665329e+09</td>
      <td>5.133841e+09</td>
      <td>34.6</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MD</td>
      <td>0.870</td>
      <td>0.850</td>
      <td>0.940</td>
      <td>0.850</td>
      <td>0.770</td>
      <td>0.875</td>
      <td>40600.0</td>
      <td>73129.0</td>
      <td>1008.0</td>
      <td>16145.69</td>
      <td>6.109971e+09</td>
      <td>1.384733e+10</td>
      <td>15.9</td>
    </tr>
    <tr>
      <th>18</th>
      <td>IA</td>
      <td>0.868</td>
      <td>0.830</td>
      <td>0.890</td>
      <td>0.827</td>
      <td>0.738</td>
      <td>0.895</td>
      <td>37166.0</td>
      <td>59113.0</td>
      <td>1075.0</td>
      <td>12346.35</td>
      <td>3.253034e+09</td>
      <td>6.216199e+09</td>
      <td>17.2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>WA</td>
      <td>0.867</td>
      <td>0.000</td>
      <td>0.931</td>
      <td>0.748</td>
      <td>0.813</td>
      <td>0.907</td>
      <td>35541.0</td>
      <td>52526.0</td>
      <td>1099.0</td>
      <td>12236.96</td>
      <td>7.833028e+09</td>
      <td>1.293234e+10</td>
      <td>16.6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>RI</td>
      <td>0.861</td>
      <td>0.770</td>
      <td>0.912</td>
      <td>0.728</td>
      <td>0.732</td>
      <td>0.905</td>
      <td>41901.0</td>
      <td>63521.0</td>
      <td>1044.0</td>
      <td>16948.19</td>
      <td>9.470488e+08</td>
      <td>2.387115e+09</td>
      <td>12.9</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ND</td>
      <td>0.859</td>
      <td>0.820</td>
      <td>0.934</td>
      <td>0.801</td>
      <td>0.829</td>
      <td>0.886</td>
      <td>35358.0</td>
      <td>55957.0</td>
      <td>1062.0</td>
      <td>14816.78</td>
      <td>8.890735e+08</td>
      <td>1.501933e+09</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>KY</td>
      <td>0.857</td>
      <td>0.730</td>
      <td>0.920</td>
      <td>0.792</td>
      <td>0.770</td>
      <td>0.884</td>
      <td>33386.0</td>
      <td>47464.0</td>
      <td>1081.0</td>
      <td>10525.46</td>
      <td>3.884563e+09</td>
      <td>7.137145e+09</td>
      <td>30.9</td>
    </tr>
    <tr>
      <th>23</th>
      <td>NE</td>
      <td>0.856</td>
      <td>0.660</td>
      <td>0.930</td>
      <td>0.800</td>
      <td>0.810</td>
      <td>0.887</td>
      <td>27274.0</td>
      <td>49999.0</td>
      <td>1039.0</td>
      <td>12773.46</td>
      <td>1.283369e+09</td>
      <td>3.930954e+09</td>
      <td>19.1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>IN</td>
      <td>0.855</td>
      <td>0.790</td>
      <td>0.935</td>
      <td>0.813</td>
      <td>0.745</td>
      <td>0.904</td>
      <td>31159.0</td>
      <td>49734.0</td>
      <td>1101.0</td>
      <td>12063.77</td>
      <td>6.764447e+09</td>
      <td>1.205895e+10</td>
      <td>23.6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>DE</td>
      <td>0.855</td>
      <td>0.000</td>
      <td>0.910</td>
      <td>0.810</td>
      <td>0.821</td>
      <td>0.884</td>
      <td>51539.0</td>
      <td>70906.0</td>
      <td>1015.0</td>
      <td>15791.15</td>
      <td>1.169017e+09</td>
      <td>1.969997e+09</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>VT</td>
      <td>0.852</td>
      <td>0.710</td>
      <td>0.870</td>
      <td>0.751</td>
      <td>0.740</td>
      <td>0.879</td>
      <td>33081.0</td>
      <td>49393.0</td>
      <td>1027.0</td>
      <td>19008.75</td>
      <td>1.532612e+09</td>
      <td>1.706096e+09</td>
      <td>10.3</td>
    </tr>
    <tr>
      <th>27</th>
      <td>TN</td>
      <td>0.839</td>
      <td>0.510</td>
      <td>0.790</td>
      <td>0.730</td>
      <td>0.770</td>
      <td>0.893</td>
      <td>29851.0</td>
      <td>39580.0</td>
      <td>1099.0</td>
      <td>9283.89</td>
      <td>4.320820e+09</td>
      <td>9.323601e+09</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>OK</td>
      <td>0.835</td>
      <td>0.700</td>
      <td>0.870</td>
      <td>0.728</td>
      <td>0.673</td>
      <td>0.877</td>
      <td>33096.0</td>
      <td>58092.0</td>
      <td>1051.0</td>
      <td>9002.55</td>
      <td>3.007448e+09</td>
      <td>6.080561e+09</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>CA</td>
      <td>0.830</td>
      <td>0.740</td>
      <td>0.929</td>
      <td>0.800</td>
      <td>0.730</td>
      <td>0.880</td>
      <td>41259.0</td>
      <td>69324.0</td>
      <td>1053.0</td>
      <td>11222.72</td>
      <td>3.929308e+10</td>
      <td>6.934292e+10</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>SC</td>
      <td>0.828</td>
      <td>0.780</td>
      <td>0.910</td>
      <td>0.790</td>
      <td>0.810</td>
      <td>0.884</td>
      <td>39196.0</td>
      <td>63474.0</td>
      <td>1042.0</td>
      <td>11524.33</td>
      <td>4.093074e+09</td>
      <td>8.640825e+09</td>
      <td>23.7</td>
    </tr>
    <tr>
      <th>31</th>
      <td>ID</td>
      <td>0.827</td>
      <td>0.720</td>
      <td>0.836</td>
      <td>0.750</td>
      <td>0.780</td>
      <td>0.820</td>
      <td>41027.0</td>
      <td>54300.0</td>
      <td>1056.0</td>
      <td>7405.57</td>
      <td>1.397871e+09</td>
      <td>2.183110e+09</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>SD</td>
      <td>0.826</td>
      <td>0.740</td>
      <td>0.940</td>
      <td>0.799</td>
      <td>0.803</td>
      <td>0.841</td>
      <td>32306.0</td>
      <td>47924.0</td>
      <td>1056.0</td>
      <td>10278.19</td>
      <td>4.189408e+08</td>
      <td>1.350969e+09</td>
      <td>25.1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MO</td>
      <td>0.823</td>
      <td>0.880</td>
      <td>0.920</td>
      <td>0.820</td>
      <td>0.789</td>
      <td>0.859</td>
      <td>30064.0</td>
      <td>47517.0</td>
      <td>1035.0</td>
      <td>11381.96</td>
      <td>3.405277e+09</td>
      <td>1.045041e+10</td>
      <td>23.4</td>
    </tr>
    <tr>
      <th>34</th>
      <td>MS</td>
      <td>0.822</td>
      <td>0.530</td>
      <td>0.836</td>
      <td>0.653</td>
      <td>0.651</td>
      <td>0.870</td>
      <td>34505.0</td>
      <td>56268.0</td>
      <td>1107.0</td>
      <td>9072.14</td>
      <td>2.244101e+09</td>
      <td>4.430399e+09</td>
      <td>32.6</td>
    </tr>
    <tr>
      <th>35</th>
      <td>OR</td>
      <td>0.816</td>
      <td>0.814</td>
      <td>0.860</td>
      <td>0.778</td>
      <td>0.771</td>
      <td>0.832</td>
      <td>31606.0</td>
      <td>44128.0</td>
      <td>1043.0</td>
      <td>11602.41</td>
      <td>3.393143e+09</td>
      <td>6.622919e+09</td>
      <td>16.6</td>
    </tr>
    <tr>
      <th>36</th>
      <td>GA</td>
      <td>0.807</td>
      <td>0.770</td>
      <td>0.916</td>
      <td>0.795</td>
      <td>0.723</td>
      <td>0.851</td>
      <td>35166.0</td>
      <td>46944.0</td>
      <td>1073.0</td>
      <td>10486.16</td>
      <td>7.918497e+09</td>
      <td>1.788841e+10</td>
      <td>23.6</td>
    </tr>
    <tr>
      <th>37</th>
      <td>NC</td>
      <td>0.804</td>
      <td>0.680</td>
      <td>0.867</td>
      <td>0.681</td>
      <td>0.685</td>
      <td>0.893</td>
      <td>31960.0</td>
      <td>46573.0</td>
      <td>1062.0</td>
      <td>9340.11</td>
      <td>8.153922e+09</td>
      <td>1.312342e+10</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>38</th>
      <td>IL</td>
      <td>0.797</td>
      <td>0.580</td>
      <td>0.800</td>
      <td>0.737</td>
      <td>0.780</td>
      <td>0.814</td>
      <td>33226.0</td>
      <td>51528.0</td>
      <td>1056.0</td>
      <td>14756.21</td>
      <td>7.088669e+09</td>
      <td>2.724015e+10</td>
      <td>18.7</td>
    </tr>
    <tr>
      <th>39</th>
      <td>MN</td>
      <td>0.797</td>
      <td>0.670</td>
      <td>0.898</td>
      <td>0.726</td>
      <td>0.674</td>
      <td>0.834</td>
      <td>35901.0</td>
      <td>61560.0</td>
      <td>1086.0</td>
      <td>13693.45</td>
      <td>8.090950e+09</td>
      <td>1.159020e+10</td>
      <td>12.6</td>
    </tr>
    <tr>
      <th>40</th>
      <td>WV</td>
      <td>0.797</td>
      <td>0.630</td>
      <td>0.000</td>
      <td>0.728</td>
      <td>0.713</td>
      <td>0.822</td>
      <td>36335.0</td>
      <td>53571.0</td>
      <td>1057.0</td>
      <td>12497.15</td>
      <td>2.074879e+09</td>
      <td>3.562152e+09</td>
      <td>29.3</td>
    </tr>
    <tr>
      <th>41</th>
      <td>AZ</td>
      <td>0.795</td>
      <td>0.677</td>
      <td>0.890</td>
      <td>0.764</td>
      <td>0.755</td>
      <td>0.840</td>
      <td>32691.0</td>
      <td>46632.0</td>
      <td>1045.0</td>
      <td>8786.17</td>
      <td>4.217359e+09</td>
      <td>9.594428e+09</td>
      <td>23.6</td>
    </tr>
    <tr>
      <th>42</th>
      <td>HI</td>
      <td>0.794</td>
      <td>0.690</td>
      <td>0.878</td>
      <td>0.734</td>
      <td>0.762</td>
      <td>0.831</td>
      <td>33664.0</td>
      <td>52880.0</td>
      <td>1080.0</td>
      <td>14434.72</td>
      <td>2.354600e+09</td>
      <td>2.696662e+09</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>43</th>
      <td>CO</td>
      <td>0.789</td>
      <td>0.620</td>
      <td>0.850</td>
      <td>0.699</td>
      <td>0.718</td>
      <td>0.844</td>
      <td>32126.0</td>
      <td>49844.0</td>
      <td>1090.0</td>
      <td>10537.69</td>
      <td>4.028316e+09</td>
      <td>9.241449e+09</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>44</th>
      <td>ME</td>
      <td>0.786</td>
      <td>0.830</td>
      <td>0.890</td>
      <td>0.730</td>
      <td>0.734</td>
      <td>0.832</td>
      <td>38655.0</td>
      <td>51381.0</td>
      <td>1011.0</td>
      <td>14604.36</td>
      <td>1.068153e+09</td>
      <td>2.670984e+09</td>
      <td>14.7</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AK</td>
      <td>0.761</td>
      <td>0.640</td>
      <td>0.810</td>
      <td>0.760</td>
      <td>0.740</td>
      <td>0.808</td>
      <td>44166.0</td>
      <td>65468.0</td>
      <td>1037.0</td>
      <td>20403.77</td>
      <td>1.835601e+09</td>
      <td>2.677359e+09</td>
      <td>25.8</td>
    </tr>
    <tr>
      <th>46</th>
      <td>PA</td>
      <td>0.748</td>
      <td>0.560</td>
      <td>0.860</td>
      <td>0.694</td>
      <td>0.660</td>
      <td>0.766</td>
      <td>33549.0</td>
      <td>58758.0</td>
      <td>1058.0</td>
      <td>17222.71</td>
      <td>1.038152e+10</td>
      <td>2.810586e+10</td>
      <td>15.8</td>
    </tr>
    <tr>
      <th>47</th>
      <td>NH</td>
      <td>0.736</td>
      <td>0.650</td>
      <td>0.860</td>
      <td>0.697</td>
      <td>0.565</td>
      <td>0.799</td>
      <td>32019.0</td>
      <td>47344.0</td>
      <td>1017.0</td>
      <td>15923.80</td>
      <td>1.005103e+09</td>
      <td>2.945559e+09</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>48</th>
      <td>NY</td>
      <td>0.710</td>
      <td>0.630</td>
      <td>0.810</td>
      <td>0.707</td>
      <td>0.610</td>
      <td>0.757</td>
      <td>48631.0</td>
      <td>68797.0</td>
      <td>1016.0</td>
      <td>23326.89</td>
      <td>2.492737e+10</td>
      <td>6.086102e+10</td>
      <td>13.2</td>
    </tr>
    <tr>
      <th>49</th>
      <td>FL</td>
      <td>0.692</td>
      <td>0.000</td>
      <td>0.770</td>
      <td>0.690</td>
      <td>0.677</td>
      <td>0.910</td>
      <td>39338.0</td>
      <td>59679.0</td>
      <td>1012.0</td>
      <td>9627.94</td>
      <td>1.046093e+10</td>
      <td>2.589709e+10</td>
      <td>19.3</td>
    </tr>
  </tbody>
</table>
</div>



### Scatter Plot of Graduation Rates Based on Race ###


```python
trace1 = go.Scatter(
    x=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
       'IA','KS','KY','LA','ME','MD','MA','MI','MN',
       'MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR',
       'PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'],
    y=grad_df['Average Rate'],
    name='Average Combined',
    yaxis='Average Combined',
    mode = 'markers'
)
trace2 = go.Scatter(
    x=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
       'IA','KS','KY','LA','ME','MD','MA','MI','MN',
       'MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR',
       'PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'],
    y=grad_df['American Indian/Alaska Native'],
    name='American Indian/Alaska Native',
    yaxis='American Indian/Alaska Native',
    mode = 'markers'
)
trace3 = go.Scatter(
    x=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
       'IA','KS','KY','LA','ME','MD','MA','MI','MN',
       'MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR',
       'PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'],
    y=grad_df['Asian/Pacific Islander'],
    name='Asian/Pacific Islander',
    yaxis='Asian/Pacific Islander',
    mode = 'markers'
)
trace4 = go.Scatter(
    x=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
       'IA','KS','KY','LA','ME','MD','MA','MI','MN',
       'MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR',
       'PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'],
    y=grad_df['Hispanic'],
    name='Hispanic',
    yaxis='Hispanic',
    mode = 'markers'
)
trace5 = go.Scatter(
    x=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
       'IA','KS','KY','LA','ME','MD','MA','MI','MN',
       'MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR',
       'PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'],
    y=grad_df['Black'],
    name='Black',
    yaxis='Black',
    mode = 'markers'
)
trace6 = go.Scatter(
    x=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
       'IA','KS','KY','LA','ME','MD','MA','MI','MN',
       'MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR',
       'PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'],
    y=grad_df['White'],
    name='White',
    yaxis='White',
    mode = 'markers'
)
data = [trace1, trace2, trace3, trace4, trace5, trace6]
layout = go.Layout(
    title='Average Graduation Rates Based on Race',
    width=800,
    xaxis=dict(
        domain=[0.1, 0.9]
    ),
    yaxis=dict(
        title='Average Combined Graduation Rate',
        titlefont=dict(
            color='#1f77b4'
        ),
        tickfont=dict(
            color='#1f77b4'
        )
    ),
    yaxis2=dict(
        title='Average American Indian/Alaska Native Graduation Rate',
        titlefont=dict(
            color='#ff7f0e'
        ),
        tickfont=dict(
            color='#ff7f0e'
        ),
        anchor='free',
        overlaying='y',
        side='left',
        position=0.15
),
    yaxis3=dict(
        title='Average Asian/Pacific Islander Graduation Rate',
        titlefont=dict(
            color='#d62728'
        ),
        tickfont=dict(
            color='#d62728'
        ),
        anchor='x',
        overlaying='y',
        side='right'
    ),
    yaxis4=dict(
        title='Average Hispanic Graduation Ratee',
        titlefont=dict(
            color='#9467bd'
        ),
        tickfont=dict(
            color='#9467bd'
        ),
        anchor='free',
        overlaying='y',
        side='right',
        position=0.85
    ),
    yaxis5=dict(
        title='Average Black Graduation Ratee',
        titlefont=dict(
            color='#4567bd'
        ),
        tickfont=dict(
            color='#4567bd'
        ),
        anchor='free',
        overlaying='y',
        side='right',
        position=0.85
    ),
    yaxis6=dict(
        title='Average White Graduation Ratee',
        titlefont=dict(
            color='#010101'
        ),
        tickfont=dict(
            color='#010101'
        ),
        anchor='free',
        overlaying='y',
        side='right',
        position=0.85
    )
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='multiple-axes-multiple')
```

### Graduation Rate v. Average Teacher Salary ###


```python
# comparing Graduation rate with Average Teacher Salary
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
y_pos = np.arange(len(education_final))
width = 0.4
education_final['Average Teacher Salary'].plot(kind='bar', color='red', ax=ax, width=width, position=1,
                                         title ="Average Graduation Rates and Average Teacher Salary per State",
                                         figsize=(20,15), fontsize=12)
education_final['Average Graduation Rate'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0, figsize=(20,15),
                                         fontsize=12)
plt.xticks(y_pos, education_final['ST'])
ax.set_ylabel('Average Teacher Salary')
ax2.set_ylabel('Average Graduation Rate')
plt.legend(loc='best')

plt.savefig('GradRates_vs_Salary.png')
plt.show()
```


![png](output_59_0.png)


### Graduation Rate vs. Average SAT Score ###


```python
# comparing Graduation rate with Average SAT Score
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
y_pos = np.arange(len(education_final))
width = 0.4
education_final['Average New SAT Score'].plot(kind='bar', color='green', ax=ax, width=width, position=1,
                                         title ="Average Graduation Rates and Average New SAT Score per State",
                                         figsize=(20,15), fontsize=12)
education_final['Average Graduation Rate'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0, figsize=(20,15),
                                         fontsize=12)
plt.xticks(y_pos, education_final['ST'])
ax.set_ylabel('Average New SAT Score')
ax2.set_ylabel('Average Graduation Rate')
plt.legend(loc='best')

plt.savefig('GradRates_vs_SAT.png')
plt.show()
```


![png](output_61_0.png)


### Graduation Rate vs. Pregnancy Rate ###


```python
# comparing Graduation rate with Pregnancy Rate
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
y_pos = np.arange(len(education_final))
width = 0.4
education_final['Pregnancy Rate (%)'].plot(kind='bar', color='orange', ax=ax, width=width, position=1,
                                         title ="Average Graduation Rates and Average Pregnancy Rates per State",
                                         figsize=(20,15), fontsize=12)
education_final['Average Graduation Rate'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0, figsize=(20,15),
                                         fontsize=12)
plt.xticks(y_pos, education_final['ST'])
ax.set_ylabel('Pregnancy Rate (%)')
ax2.set_ylabel('Average Graduation Rate')
plt.legend(loc='best')

plt.savefig('GradRates_vs_PregRate.png')
plt.show()
```


![png](output_63_0.png)


### Gradution Rates vs. Spending per Student ###


```python
# comparing Graduation Rates with Spending per Student
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
y_pos = np.arange(len(education_final))
width = 0.4
education_final['Spending per Student'].plot(kind='bar', color='gold', ax=ax, width=width, position=1,
                                         title ="Average Graduation Rates and Spending per Student by State",
                                         figsize=(20,15), fontsize=12)
education_final['Average Graduation Rate'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0, figsize=(20,15),
                                         fontsize=12)
plt.xticks(y_pos, education_final['ST'])
ax.set_ylabel('Spending per Student')
ax2.set_ylabel('Average Graduation Rate')
plt.legend(loc='best')

plt.savefig('GradRates_vs_StuSpend.png')
plt.show()
```


![png](output_65_0.png)


### Data and Resources Used ###

#### SAT Scores per State: ####
https://blog.prepscholar.com/average-sat-and-act-scores-by-stated-adjusted-for-participation-rate

#### Overall Teacher Salaries per state: ####
https://articles.niche.com/teacher-salaries-in-america/
http://www.teacherportal.com/teacher-salaries-by-state/

#### Graduation rates by state 2014-2015: ####
https://nces.ed.gov/programs/coe/indicator_coi.asp

#### Graduation by Household Income: ####
 http://gradnation.americaspromise.org/sites/default/files/d8/2017-05/Appendix_G.pdf

#### Teen Pregnancy Rates: ####
https://www.cdc.gov/nchs/pressroom/sosmap/teen-births/teenbirths.html

#### Spending per Student: ####
https://www.census.gov/data/tables/2014/econ/school-finances/secondary-education-finance.html

