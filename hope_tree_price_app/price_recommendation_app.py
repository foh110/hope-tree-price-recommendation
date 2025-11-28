import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 1. 加载模型（不变） --------------------------
def load_all_models(model_dir='./hope_tree_models'):
    if not Path(model_dir).exists():
        st.error(f"模型目录 {model_dir} 不存在，请先运行多表分析代码！")
        return None
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
    if len(model_files) == 0:
        st.error(f"模型目录 {model_dir} 无有效模型！")
        return None
    all_models = {}
    for file in model_files:
        sheet_name = file.replace('_model.joblib', '')
        model_info = joblib.load(os.path.join(model_dir, file))
        if 'fixed_rsp' not in model_info:
            st.warning(f"店铺 {sheet_name} 缺少固定RSP数据，暂不支持")
            continue
        all_models[sheet_name] = model_info
    return all_models

# -------------------------- 2. 折扣率计算（核心修改：公式替换） --------------------------
def calculate_discount_by_formula(input_price, fixed_rsp):
    """正确公式：折扣率 = 1 - (销售输入罐单 / 店铺固定RSP)"""
    try:
        discount_rate = 1 - (input_price / fixed_rsp)  # 正确公式
        if discount_rate < 0:
            return None, f"错误：输入罐单（{input_price:.2f}元）超过RSP（{fixed_rsp:.2f}元），折扣率为负"
        return discount_rate, f"折扣率计算完成（公式：1 - (输入罐单/RSP)）"
    except Exception as e:
        return None, f"计算错误：{str(e)}"

# -------------------------- 3. 预测函数（不变） --------------------------
def predict_by_model(model_info, calculated_discount):
    gam_sales = model_info['gam_sales']
    gam_returns = model_info['gam_returns']
    # 确保折扣率在训练范围内（原逻辑不变）
    discount = np.clip(calculated_discount, model_info['discount_min'], model_info['discount_max'])
    if not (model_info['discount_min'] <= calculated_discount <= model_info['discount_max']):
        st.warning(f"计算的折扣率（{calculated_discount:.2%}）超出模型训练范围（{model_info['discount_min']:.2%}~{model_info['discount_max']:.2%}），预测结果可能存在偏差")

    # 核心修复：销量预测结果强制非负（加max(..., 0)）
    pred_sales = max(gam_sales.predict([[discount]])[0], 0)  # 修复后：负数转为0
    pred_returns = max(gam_returns.predict([[discount]])[0], 0)  # 退款率已有约束，保留

    # 置信区间也同步修正（避免置信区间下限为负）
    sales_std = model_info['sales_metrics']['residual_std']
    sales_ci_lower = max(pred_sales - 1.96 * sales_std, 0)  # 置信区间下限≥0
    sales_ci_upper = pred_sales + 1.96 * sales_std
    returns_std = model_info['returns_metrics']['residual_std']
    returns_ci = (max(pred_returns - 1.96 * returns_std, 0), min(pred_returns + 1.96 * returns_std, 1))

    return {
        'discount': discount,
        'pred_sales': round(pred_sales),  # 此时销量≥0，四舍五入后仍是非负整数
        'sales_ci': (round(sales_ci_lower), round(sales_ci_upper)),
        'pred_returns': round(pred_returns, 4),
        'returns_ci': (round(returns_ci[0], 4), round(returns_ci[1], 4))
    }

# -------------------------- 4. 可视化图表（不变） --------------------------
def plot_discount_impact(model_info, calculated_discount):
    gam_sales = model_info['gam_sales']
    gam_returns = model_info['gam_returns']
    discount_range = np.linspace(model_info['discount_min'], model_info['discount_max'], 200)
    pred_sales = gam_sales.predict(discount_range.reshape(-1, 1))
    pred_returns = np.maximum(gam_returns.predict(discount_range.reshape(-1, 1)), 0)
    optimal_discount = model_info['optimal_discount']
    optimal_sales = gam_sales.predict([[optimal_discount]])[0]
    optimal_returns = max(gam_returns.predict([[optimal_discount]])[0], 0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 销量轴
    color_sales = '#2E86AB'
    ax1.plot(discount_range, pred_sales, color=color_sales, linewidth=2, label='销量预测')
    ax1.scatter(optimal_discount, optimal_sales, color=color_sales, s=100, zorder=5, label=f'模型最优折扣点({optimal_discount:.2%})')
    ax1.scatter(calculated_discount, predict_by_model(model_info, calculated_discount)['pred_sales'], color='red', s=100, zorder=5, label=f'公式计算折扣点({calculated_discount:.2%})')
    ax1.set_xlabel('折扣率', fontsize=12, fontweight='bold')
    ax1.set_ylabel('预测销量（件）', color=color_sales, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_sales)
    ax1.grid(alpha=0.3)
    # 退款率轴
    ax2 = ax1.twinx()
    color_returns = '#A23B72'
    ax2.plot(discount_range, pred_returns, color=color_returns, linewidth=2, linestyle='--', label='退款率预测')
    ax2.scatter(optimal_discount, optimal_returns, color=color_returns, s=100, zorder=5)
    ax2.scatter(calculated_discount, predict_by_model(model_info, calculated_discount)['pred_returns'], color='red', s=100, zorder=5)
    ax2.set_ylabel('预测退款率', color=color_returns, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_returns)
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    # 标题
    plt.title(f'折扣率对销量和退款率的影响（店铺固定RSP：{model_info["fixed_rsp"]:.2f}元）', fontsize=14, fontweight='bold', pad=15)
    return fig

# -------------------------- 5. 网页主界面（仅更新文本说明） --------------------------
def main():
    # 1. 背景图片设置（已适配你的路径）
    # 处理后的路径：去掉开头隐藏字符，将单反斜杠改为双反斜杠
    # background_image_path = "image.jpg"  # 直接复制这行即可
    #
    # st.markdown(
    #     f"""
    #     <style>
    #     .stApp {{
    #         background-image: url("{background_image_path}");
    #         background-size: cover;
    #         background-repeat: no-repeat;
    #         background-attachment: fixed;
    #         background-position: center;
    #         opacity: 0.95;
    #         position: relative;
    #     }}
    #     /* 半透明白色遮罩，避免文字与背景冲突 */
    #     .stApp::before {{
    #         content: "";
    #         position: absolute;
    #         top: 0;
    #         left: 0;
    #         width: 100%;
    #         height: 100%;
    #         background-color: rgba(255, 255, 255, 0.8);
    #         z-index: -1;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    # 2. 标题+备注（原逻辑不变）
    st.title("希望树商品价格推荐系统")
    st.subheader("—— 基于RSP约束与公式计算的折扣率预测")
    st.warning("⚠️ 为保证数据安全，仅公司wifi可开启")
    # -------------------------- 新增：备注+链接 --------------------------
    # 飞书链接
    your_link = "https://bestage.feishu.cn/wiki/UtOtwqpEXi8kR2kUVD5c75jHnKc"
    # 备注文字+嵌入链接
    st.markdown(
        f"""
        - 本模型基于抖音两个直播间、天猫官旗 top销售sku近一年的销售数据，采用GAM模型得出。具体细节请参考
          <a href="{your_link}" target="_blank" style="color: #1E40AF; text-decoration: underline;">
              价格变动敏感度分析
          </a>
        """,
        unsafe_allow_html=True
    )

    # 加载模型（原逻辑不变）
    with st.spinner("正在加载模型..."):
        all_models = load_all_models(model_dir='./hope_tree_models')
    if all_models is None:
        return

    # 输入区域（原逻辑不变，新增固定成本的获取）
    st.sidebar.header("输入参数")
    selected_sheet = st.sidebar.selectbox("选择店铺/渠道", options=list(all_models.keys()))
    model_info = all_models[selected_sheet]
    fixed_rsp = model_info['fixed_rsp']
    fixed_cost = model_info['fixed_cost']  # 新增：获取店铺固定成本
    st.sidebar.info(f"当前店铺固定RSP：{fixed_rsp:.2f}元 | 固定成本：{fixed_cost:.2f}元")
    st.sidebar.warning("⚠️ 输入罐单价需≤RSP，否则折扣率为负")
    recommended_price = fixed_rsp * (1 - model_info['optimal_discount'])
    # 输入罐单价（原逻辑不变）
    input_price = st.sidebar.number_input(
        "输入罐单价（元）",
        min_value=0.01,
        max_value=fixed_rsp - 0.01,
        value=min(round(fixed_rsp * 0.8, 2), fixed_rsp - 0.01),
        step=0.01
    )

    # 计算折扣率（原逻辑不变）
    calculated_discount, calc_msg = calculate_discount_by_formula(input_price, fixed_rsp)
    if calculated_discount is None:
        st.sidebar.error(calc_msg)
        return
    st.sidebar.success(calc_msg)
    st.sidebar.metric("公式计算折扣率", f"{calculated_discount:.2%}")

    # 预测销量/退款率（原逻辑不变）
    predict_result = predict_by_model(model_info, calculated_discount)

    # -------------------------- 关键修改：结果展示区（实现3个需求） --------------------------
    st.markdown(
        f"## 📊 推荐结果（模型推荐罐单：<span class='recommend-price'>{recommended_price:.2f}元</span>）",
        unsafe_allow_html=True  # 该参数在st.markdown()中是合法的
    )

    # 下方指标排列完全不变（保持原布局）
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "公式计算折扣率",
        f"{predict_result['discount']:.2%}",
        help=f"计算逻辑：1 - ({input_price:.2f}/{fixed_rsp:.2f})"
    )
    col2.metric(
        "输入罐单价（元）",
        f"{input_price:.2f}",
        help="你输入的罐单价"
    )
    col3.metric(
        "预测销量（件）",
        f"{predict_result['pred_sales']}",
        help=f"95%置信区间：{predict_result['sales_ci'][0]}~{predict_result['sales_ci'][1]}件"
    )

    col4, col5 = st.columns(2)
    col4.metric(
        "预测退款率",
        f"{predict_result['pred_returns']:.2%}",
        help=f"95%置信区间：{predict_result['returns_ci'][0]:.2%}~{predict_result['returns_ci'][1]:.2%}"
    )

    if input_price <= fixed_cost:
        gross_profit_rate = 0
        col5.metric(
            "预估毛利率",
            f"{gross_profit_rate:.2f}%",
            help=f"输入罐单价（{input_price:.2f}元）≤固定成本（{fixed_cost:.2f}元），毛利为负"
        )
        st.warning(f"⚠️ 输入罐单价低于/等于固定成本，无盈利空间！")
    else:
        gross_profit_rate = (input_price - fixed_cost) / input_price * 100
        col5.metric(
            "预估毛利率",
            f"{gross_profit_rate:.2f}%",
            help=f"计算逻辑：({input_price:.2f}-{fixed_cost:.2f})/{input_price:.2f} × 100%"
        )

    # 后续的模型可信度、可视化、业务建议（不变）
    st.header("🔍 模型可信度")
    sales_adj_r2 = model_info['sales_metrics']['adj_r2']
    returns_adj_r2 = model_info['returns_metrics']['adj_r2']
    reliability = '高' if (sales_adj_r2 >= 0.3 and returns_adj_r2 >= 0.3) else '中' if (
                sales_adj_r2 >= 0.1 or returns_adj_r2 >= 0.1) else '低'
    st.write(f"• 店铺/渠道：{selected_sheet}（固定RSP：{fixed_rsp:.2f}元 | 固定成本：{fixed_cost:.2f}元）")
    st.write(f"• 销量模型调整后R²：{sales_adj_r2:.4f}")
    st.write(f"• 退款率模型调整后R²：{returns_adj_r2:.4f}")
    st.write(f"• 综合可信度：{reliability}")

    st.header("📈 趋势可视化")
    fig = plot_discount_impact(model_info, calculated_discount)
    st.pyplot(fig)

    st.header("💡 业务建议")
    if reliability == '高':
        st.success("• 模型预测可靠，可直接按输入罐单价执行；\n• 建议跟踪实际销量/退款率，验证公式计算的折扣率是否准确。")
    elif reliability == '中':
        st.warning("• 模型有参考价值，建议先小范围测试（如1天）；\n• 测试时重点关注“输入罐单价-折扣率-销量”的匹配度。")
    else:
        st.error("• 模型解释力有限，需结合行业经验调整罐单价；\n• 优先补充该店铺的“罐单价-RSP-销量”历史数据。")


if __name__ == "__main__":
    main()