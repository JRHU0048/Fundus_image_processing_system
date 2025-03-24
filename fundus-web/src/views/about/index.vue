<template>
  <div class="app-container">
    <h1 class="page-title">眼底图像处理系统介绍</h1>
    <h2 class="project-overview-title">一、项目概述</h2>
    <p class="project-overview-desc">本项目提供眼底图像分类方案，助医疗人员精准诊断眼病。</p>

    <h2 class="feature-accordion-title">二、功能特点</h2>
    <div>
      <h3>1、多疾病分类</h3>
      <p class="feature-description">能同时分类多种眼病，提升诊断效率。</p>
      <h3>2、高精度模型</h3>
      <p class="feature-description">经大量数据训练，模型准确率和召回率高。</p>
    </div>

    <h2 class="technical-implementation-title">三、技术实现</h2>
    <div>
      <h3>1、图像分类</h3>
      <p class="technical-description">用深度学习算法（CNN）提取分类眼底图像特征。</p>
      <h3>2、高性能设备</h3>
      <p class="technical-description">借助高性能设备加速模型训练与推理。</p>
      <h3>3、用户界面</h3>
      <p class="technical-description">打造友好界面方便操作查看结果。</p>
    </div>

    <h2 class="application-scenarios-title">四、应用场景</h2>
    <div>
      <h3>1、眼科诊所</h3>
      <p class="scenario-description">助力眼科医生快速诊断患者眼病。</p>
      <h3>2、医学研究</h3>
      <p class="scenario-description">为医学研究提供数据支持，推动眼病研究发展。</p>
    </div>
  </div>
</template>

<script>
export default {};
</script>

<style scoped>
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}
.page-title {
  text-align: center;
  margin-bottom: 20px;
  font-size: 32px;
  color: #333;
}
.project-overview-title,
.feature-accordion-title,
.technical-implementation-title,
.application-scenarios-title {
  margin-top: 30px;
  margin-bottom: 10px;
  font-size: 24px;
  color: #333;
}
h3 {
  margin-bottom: 5px;
  color: #333;
}
.project-overview-desc,
.feature-description,
.scenario-description,
.technical-description {
  color: #666;
  line-height: 1.6;
  text-indent: 2em;
}
</style>