import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 页面设置
st.set_page_config(page_title="数据分析Web应用", layout="wide")
st.title("数据分析工具")

# 上传数据
uploaded_file = st.sidebar.file_uploader("选择一个数据文件", type=["csv", "xlsx", "json"])

# 加载数据
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        return pd.read_json(uploaded_file)
    else:
        st.error("不支持的文件格式")
        return None

# 单元格操作函数
def operation_cell(df, cell_id):
    st.write(f"操作单元格 {cell_id + 1}")
    operation = st.selectbox("选择操作", ["数据清洗", "可视化", "K-Means 聚类", "主成分分析","二分类Logistic回归"], key=f"operation_{cell_id}")

    # 用于保存操作结果的字典
    if f"output_{cell_id}" not in st.session_state:
        st.session_state[f"output_{cell_id}"] = []

    if operation == "数据清洗":
        clean_option = st.selectbox("清洗选项", ["删除缺失值行", "填充缺失值", "删除重复行"], key=f"clean_{cell_id}")
        if clean_option == "删除缺失值行" and st.button("执行操作", key=f"drop_na_{cell_id}"):
            df = df.dropna()
            st.success("已删除所有包含缺失值的行。")
            st.session_state[f"output_{cell_id}"].append(("dataframe", df.head()))

        elif clean_option == "填充缺失值":
            fill_column = st.selectbox("选择要填充缺失值的列", options=df.columns, key=f"fill_column_{cell_id}")
            fill_value = st.text_input("填充值", "", key=f"fill_value_{cell_id}")
            if st.button("执行操作", key=f"fill_na_{cell_id}") and fill_value:
                df[fill_column] = df[fill_column].fillna(fill_value)
                st.success(f"已将列 `{fill_column}` 的缺失值填充为 `{fill_value}`。")
                st.session_state[f"output_{cell_id}"].append(("dataframe", df.head()))

        elif clean_option == "删除重复行" and st.button("执行操作", key=f"drop_duplicates_{cell_id}"):
            df = df.drop_duplicates()
            st.success("已删除重复行。")
            st.session_state[f"output_{cell_id}"].append(("dataframe", df.head()))

    elif operation == "可视化":
        plot_type = st.selectbox("选择图表类型", ["散点图", "折线图", "饼图", "箱线图"], key=f"plot_type_{cell_id}")
        x_axis = st.selectbox("选择X轴", df.columns, key=f"x_axis_{cell_id}")
        y_axis = st.selectbox("选择Y轴", df.columns, key=f"y_axis_{cell_id}")
        hue = st.selectbox("颜色分类（可选）", [None] + list(df.columns), key=f"hue_{cell_id}")

        if st.button("生成图表", key=f"plot_btn_{cell_id}"):
            if plot_type == "散点图":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=hue)
            elif plot_type == "折线图":
                fig = px.line(df, x=x_axis, y=y_axis, color=hue)
            elif plot_type == "饼图":
                fig = px.pie(df, names=hue, title=f"{hue} 分布")
            elif plot_type == "箱线图":
                fig = px.box(df, x=hue, y=y_axis)

            st.session_state[f"output_{cell_id}"].append(("chart", fig))

    elif operation == "K-Means 聚类":
        st.subheader("K-Means 聚类")

        # 获取数值型的列供用户选择
        numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
        selected_columns = st.multiselect("选择用于聚类的列", numeric_columns, key=f"cluster_columns_{cell_id}")
        n_clusters = st.number_input("选择簇数", min_value=2, max_value=10, value=3, step=1, key=f"clusters_{cell_id}")

        # 距离度量选择
        distance_metric = st.selectbox("选择距离度量", ("欧式距离", "曼哈顿距离"), key=f"distance_metric_{cell_id}")

        def calculate_distance(data, metric):
            if metric == "欧式距离":
                return data
            elif metric == "曼哈顿距离":
                return np.abs(data - data.mean())

        if st.button("显示肘部图", key=f"elbow_{cell_id}") and selected_columns:
            sse = []
            transformed_data = calculate_distance(df[selected_columns], distance_metric)
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(transformed_data)
                sse.append(kmeans.inertia_)

            fig = px.line(x=range(1, 11), y=sse, labels={'x': '簇数', 'y': '簇内误差平方和 (SSE)'})
            fig.update_traces(mode='lines+markers')
            st.session_state[f"output_{cell_id}"].append(("chart", fig))

        if st.button("执行聚类", key=f"cluster_{cell_id}") and selected_columns:
            try:
                transformed_data = calculate_distance(df[selected_columns], distance_metric)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                df['Cluster'] = kmeans.fit_predict(transformed_data)

                # 存储聚类结果
                st.session_state[f"output_{cell_id}"].append(("dataframe", df[['Cluster'] + selected_columns].head()))

                # 存储簇的描述性统计
                cluster_description = df.groupby('Cluster')[selected_columns].describe()
                st.session_state[f"output_{cell_id}"].append(("dataframe", cluster_description))
            except Exception as e:
                st.error(f"聚类时出错: {e}")

    elif operation == "主成分分析":
        st.subheader("主成分分析 (PCA)")

        # 选择数值列用于主成分分析
        numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
        selected_columns = st.multiselect("选择用于主成分分析的列", numeric_columns, key=f"pca_columns_{cell_id}")

        if selected_columns:
            n_components = st.number_input("选择主成分数量", min_value=1, max_value=len(selected_columns), value=2, step=1, key=f"pca_components_{cell_id}")
            if st.button("执行PCA", key=f"perform_pca_{cell_id}"):
                try:
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(df[selected_columns])

                    # 保存PCA结果
                    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i + 1}" for i in range(n_components)])
                    st.session_state[f"output_{cell_id}"].append(("dataframe", pca_df.head()))

                    # 显示方差贡献率
                    explained_variance = pca.explained_variance_ratio_ * 100
                    variance_df = pd.DataFrame(
                        {"主成分": [f"PC{i + 1}" for i in range(n_components)], "方差贡献率 (%)": explained_variance})
                    st.write("各主成分的方差贡献率：")
                    st.session_state[f"output_{cell_id}"].append(("dataframe", variance_df))

                    # 如果有两个或更多主成分，绘制散点图
                    if n_components >= 2:
                        fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA 散点图 (前两个主成分)", labels={"PC1": "主成分 1", "PC2": "主成分 2"})
                        st.session_state[f"output_{cell_id}"].append(("chart", fig))
                except Exception as e:
                    st.error(f"主成分分析时出错: {e}")
        else:
            st.info("请选择至少一列用于主成分分析")

    elif operation == "二分类Logistic回归":
        st.subheader("二分类Logistic回归")

        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        from scipy.stats import norm
        import numpy as np

        # 选择用于回归的列
        columns = df.select_dtypes(include=['float64', 'int', 'string']).columns.tolist()
        dependent_variable = st.multiselect("选择因变量", columns, key=f"dependent_variable_{cell_id}")
        if len(dependent_variable)!=1:
            st.info("请选择一列作为因变量")
        independent_variable = st.multiselect("选择自变量", columns, key=f"independent_variable_{cell_id}")
        df[dependent_variable]=df[dependent_variable].astype('float')
        df[independent_variable]=df[independent_variable].astype('float')
        x = '+'.join(independent_variable)
        y = dependent_variable[0]

        if independent_variable:

            #模型拟合
            if st.button("执行二分类Logistic回归", key=f"train_{cell_id}"):
                try:
                    def logistic_regression(y,x,df):
                        model = smf.glm(formula = f'{y} ~ {x}',
                                        data = df,
                                        family=sm.families.Binomial()).fit()  
                        results_as_html = model.summary().tables[1].as_html()
                        result = pd.read_html(results_as_html, header=0, index_col=0)[0]
                        st.session_state[f"output_{cell_id}"].append(("dataframe", result))
                    logistic_regression(y=y,x=x,df=df)
                except Exception as e:
                    st.error(f"Logistic回归时出错: {e}")

            #计算OR值
            if st.button("计算OR值", key=f"OR_{cell_id}"):
                try:
                    def OR(y,x,df):
                        model = smf.glm(formula = f'{y} ~ {x}',
                                        data = df,
                                        family=sm.families.Binomial()).fit()  
                        stat = pd.DataFrame({'p': model.pvalues,                      
                                             'OR': np.exp(model.params), 
                                            'OR_lower_ci': np.exp(model.params - norm.ppf(0.975)*model.bse),
                                            'OR_upper_ci': np.exp(model.params + norm.ppf(0.975)*model.bse)}) 
                        stat['sig'] = stat.apply(lambda x : "*" if x['p']<0.05 else "no_sig",axis=1)
                        stat= stat.sort_values('OR', ascending=True)                            
                        st.session_state[f"output_{cell_id}"].append(("dataframe", stat))
                    OR(y=y,x=x,df=df)
                except Exception as e:
                    st.error(f"计算OR值时出错: {e}")

            #绘制OR森林图
            if st.button("绘制OR森林图", key=f"OR_plot_{cell_id}"):
                try:
                    def OR_plot(y,x,df):
                        model = smf.glm(formula = f'{y} ~ {x}',
                                        data = df,
                                        family=sm.families.Binomial()).fit()  
                        stat = pd.DataFrame({'p': model.pvalues,                      
                                             'OR': np.exp(model.params), 
                                            'OR_lower_ci': np.exp(model.params - norm.ppf(0.975)*model.bse),
                                            'OR_upper_ci': np.exp(model.params + norm.ppf(0.975)*model.bse)}) 
                        stat['sig'] = stat.apply(lambda x : "*" if x['p']<0.05 else "no_sig",axis=1)
                        stat= stat.sort_values('OR', ascending=True)                            
                        forest_df = stat.drop("Intercept")\
                                        .reset_index()\
                                        .rename(columns={'index': 'independent_var'})\
                                        .sort_values('OR', ascending=False)
                        from plotnine import ggplot,aes,geom_point,geom_errorbarh,scale_color_manual,scale_y_discrete,guides,guide_legend,labs,geom_vline,theme_minimal,theme,element_text
                        forest = ggplot(forest_df , 
                                        aes(y='independent_var', x='OR')) + geom_point(aes(color='sig'),size=2) + geom_errorbarh(aes(xmin='OR_lower_ci', xmax='OR_upper_ci',color ='sig'), height=0.1) + scale_color_manual(values = ["red","black"]) + scale_y_discrete(limits= forest_df["independent_var"]) + guides(color=guide_legend(reverse=True))+labs(title='logistic Regression', x='OR', y='variable')+geom_vline(xintercept=1, linetype='dashed', color='black')+theme_minimal()+theme(plot_title=element_text(hjust=0.5))
                        st.pyplot(ggplot.draw(forest)) 
                    OR_plot(y=y,x=x,df=df)
                except Exception as e:
                    st.error(f"绘制OR森林图时出错: {e}")

        else:
            st.info("请选择至少一列作为自变量")
        

    # 显示已保存的结果
    for result_type, content in st.session_state[f"output_{cell_id}"]:
        if result_type == "dataframe":
            st.dataframe(content)
        elif result_type == "chart":
            st.plotly_chart(content, use_container_width=True)

    # 添加单元格按钮放置在每个单元格操作之后
    if st.button("添加单元格", key=f"add_cell_{cell_id}"):
        st.session_state['num_cells'] += 1

# 初始化单元格计数
if 'num_cells' not in st.session_state:
    st.session_state['num_cells'] = 1

# 主逻辑
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("数据预览")
        st.dataframe(df.head())

        # 根据当前单元格计数生成单元格
        for cell_id in range(st.session_state['num_cells']):
            operation_cell(df, cell_id)
else:
    st.info("请通过左侧边栏上传一个数据文件")
