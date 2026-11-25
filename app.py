import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
st.set_page_config(page_title='Diagnóstico econométrico', layout='wide')
st.title('E-commerce | Diagnóstico econométrico')
st.caption('Entregable 4 · Regresión múltiple')
d = pd.read_csv(Path(__file__).with_name('datos.csv'))
x = sm.add_constant(d[['Quantity','Discount(%)']]); y = d['Sales_Amount']
m = sm.OLS(y,x).fit(); r=m.get_robustcov_results(cov_type='HC3')
jb,pj,_,_=jarque_bera(m.resid); lm,pb,_,_=het_breuschpagan(m.resid,x)
vif=[variance_inflation_factor(x.values,i) for i in (1,2)]
a,b,c=st.columns(3); a.metric('Registros',len(d)); b.metric('R²',f'{m.rsquared:.4f}'); c.metric('VIF máximo',f'{max(vif):.3f}')
st.subheader('Pruebas de diagnóstico (α = 0.05)')
st.dataframe(pd.DataFrame({'Prueba':['Jarque-Bera','Breusch-Pagan','VIF cantidad','VIF descuento'],'Estadístico':[jb,lm,*vif],'p-valor':[pj,pb,None,None],'Interpretación':['No normalidad' if pj<.05 else 'No se rechaza normalidad','Heterocedasticidad' if pb<.05 else 'No se rechaza homocedasticidad',*['VIF bajo' if v<5 else 'Revisar VIF' for v in vif]]}),hide_index=True)
st.subheader('Residuos')
st.scatter_chart(pd.DataFrame({'Venta estimada':m.fittedvalues,'Residuo':m.resid}),x='Venta estimada',y='Residuo')
st.subheader('Coeficientes con errores robustos HC3')
st.dataframe(pd.DataFrame({'Variable':x.columns,'Coeficiente':r.params,'Error robusto':r.bse,'p-valor':r.pvalues}),hide_index=True)
st.subheader('Simulación de ventas al 95 %')
q=st.slider('Cantidad',1,5,3); discount=st.slider('Descuento (%)',0,30,15)
p=r.get_prediction(pd.DataFrame({'const':[1.0],'Quantity':[q],'Discount(%)':[discount/100]})).summary_frame(alpha=.05).iloc[0]
st.metric('Venta media estimada',f'{p["mean"]:,.2f}')
st.write(f'Intervalo de confianza de la media: {p["mean_ci_lower"]:,.2f} a {p["mean_ci_upper"]:,.2f}')
st.write(f'Intervalo de predicción individual aproximado: {p["obs_ci_lower"]:,.2f} a {p["obs_ci_upper"]:,.2f}')
st.caption('El intervalo individual convencional puede no tener cobertura adecuada bajo heterocedasticidad o no normalidad. HC3 corrige la inferencia de coeficientes, no la distribución de residuos.')
with st.expander('Datos utilizados'): st.dataframe(d)
