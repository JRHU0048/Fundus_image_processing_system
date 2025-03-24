<template>
  <div class="dashboard-container">

    <div class="introduction-text">
      <p>本模块专为眼底图像分类任务，能够提供准确的眼底图像多疾病分类结果</p>
    </div>

    <div class="container-photo">
      <div class="left-option">
        <el-form>
          <el-form-item label="模型选择">
            <el-select v-model="modules" placeholder="请选择合适的模型">
              <!-- 修改：调整选项值与后端对应 -->
              <el-option label="多疾病分类模型" value="fundus_classifier" />
              <el-option label="青光眼分级模型" value="glu_classify_mode" />
            </el-select>
          </el-form-item>
          <el-form-item style="margin-left:68px;">
            <el-button type="primary" style="width:200px;" @click="onSubmit">立即检测</el-button>
          </el-form-item>
        </el-form>
      </div>

      <div class="right-present">
        <div v-if="photoUrl==''||photoUrl==null">
          <el-upload
            class="upload-demo"
            drag
            action="http://localhost:81/api/file/photo"
            :on-preview="handlePreview"
            :on-remove="handleRemove"
            :on-success="handleSuccess"
            :before-upload="handleBefore"
          >
            <i class="el-icon-upload" />
            <div class="el-upload__text">请将<em>分类</em>图片拖到此处，或<em>点击上传</em></div>
            <div slot="tip" class="el-upload__tip">目前系统仅支持jpg/png格式的图片</div>
          </el-upload>
        </div>

        <!-- 虚线框布局 -->
        <!-- <div v-else style="display: flex; flex-direction: row;"> -->
        <div v-else style="display: flex; flex-direction: row; align-items: center; gap: 10px;">

          <!-- 图片和分类结果上下布局 -->
          <div style="display: flex; flex-direction: column; gap: 5px;">
            <!-- 原始图片展示区域 -->
            <div>
              <img :src="photoUrlBefore" alt="Original" style="width: 250px; height: 250px;"/>
            </div>
            <!-- 分类结果展示区域 -->
            <el-card v-if="classificationResult" class="result-card" style="width: 250px;">
              <div slot="header" class="clearfix">
                <span style="font-size: 18px;">诊断建议</span>
              </div>
              <!-- 根据后端返回结构调整遍历 -->
              <div v-for="(result, index) in classificationResult.predictions" :key="index" class="result-item">
                <span class="disease-name">{{ result.disease }}</span>
                <el-progress 
                  :percentage="(result.confidence * 100).toFixed(2)" 
                  :status="result.confidence > 0.7 ? 'success' : 'warning'"
                  :stroke-width="16"
                />
                <span class="confidence-value">{{ (result.confidence * 100).toFixed(2) }}%</span>
              </div>
              <!-- 新增：显示 top_class -->
              <div class="top-class">
                <span style="font-size: 16px; color: #606266;">可能疾病: {{ classificationResult.top_class }}</span>
              </div>
            </el-card>
          </div>
        
          <!-- 新增专家分析卡片 -->
          <el-card class="expert-card">
            <div slot="header" class="clearfix">
              <span style="font-size: 18px;">AI专家分析报告</span>
            </div>
            <div class="expert-content">
              <div class="analysis-item">
                <h3>📌 病理解读</h3>
                <p>{{ classificationResult.expert_analysis.pathological_interpretation }}</p>
              </div>
              <div class="analysis-item">
                <h3>💡 治疗建议</h3>
                <p>{{ classificationResult.expert_analysis.treatment_recommendation }}</p>
              </div>
            </div>
          </el-card>

        </div>

        <el-button v-if="photoUrl!=null&&photoUrl!=''" type="primary" style="width:200px; margin-top:5px" @click="onBack">重新上传</el-button>
      </div>
    </div>
  </div>
</template>


<script>

export default {
  data: function() {
    return {
      modules: '',  // 存储选择的模型
      photoUrl: '',   // 存储上传图片的 URL
      photoUrlBefore: '', // 专门存储原始图片URL
      // 修改：添加分类结果存储变量
      classificationResult: null,
      imageName: ''  // 存储上传图片的名称
    }
  },
  methods: {
    onSubmit() {
      const that = this // 保存当前组件实例
      const loading = this.$loading({  // 显示加载动画
        lock: true,
        text: '检测中...',
        spinner: 'el-icon-loading',
        background: 'rgba(0, 0, 0, 0.7)'
      })

      // 发起请求，调用 req 方法
      this.req({
        url: '/file/checkphoto',
        method: 'get',
        params: {
          model: that.modules,
          imageName: that.imageName
        }
      }).then(res => {
        console.log("前端接收到的响应:", res);
        console.log("res.data 的格式:", res.data);
        console.log(res)
        // 修改：根据后端返回结构调整赋值
        if (res.data.msg === 'ok') {
          this.photoUrl = this.photoUrlBefore; // 保进入结果展示区块
          this.classificationResult = res.data;
          console.log("classificationResult 的格式:", this.classificationResult);
        } else {
          this.$message.error(res.data.error);
        }
        loading.close()
      }).catch(error => {
        this.$message.error('请求出错，请稍后重试');
        loading.close();
      });
    },
    onBack() {  // 重置结果
      this.photoUrl = ''
      this.photoUrlBefore = ''
      this.classificationResult = null
    },
    handleBefore() {
    },
    submitUpload() {
      this.$refs.upload.submit()
    },
    handleRemove(file, fileList) {
      console.log(file, fileList)
    },
    handlePreview(file) {
      console.log(file)
    },
    handleSuccess(res, file, fileList) {
      this.photoUrlBefore = res.imageUrl // 原始图片URL
      // this.photoUrl = res.imageUrl
      this.imageName = res.imageName
      console.log('res', res)
    }
  }
}
</script>


<style>
/* 新增样式 */
.result-card {
  background: #f8fafc;
  width: 90%;
  margin-top: 0px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,.1);
}

.result-item {
  margin: 10px 0;
  display: flex;
  align-items: center;
}

.disease-name {
  width: 120px;
  font-size: 16px;
  color: #606266;
}

.confidence-value {
  margin-left: 15px;
  color: #67C23A;
  font-weight: bold;
}

.re-upload-btn {
  width: 200px;
  margin-top: 30px;
}

.top-class {
  margin-top: 15px;
  font-size: 16px;
  color: #606266;
}
/* 新增样式 */

.introduction-text {
  position: center;
  width: 100%;
  padding: 20px;
  background-color: #f0f8ff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, .1);
  margin-bottom: 30px;
  text-align: center;

  p {
    font-size: 18px;
    color: #333;
    line-height: 1.5;
  }
}

.container-photo {
  display: flex;
  flex-direction: row;
  height: 60vh;
  width: 80%;
  /* max-width: 1200px;  */
  margin: 0 auto;
  /* align-items: center; */

  .left-option {
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex: 1;
  }

  .right-present {
    flex: 2;
    border: dotted #000000;
    border-radius: 20px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 70vh; /* 将高度增加到70vh*/
    width: 90%; /* 将宽度增加到90%*/
    /* height: auto; 让高度自动适应内容 */
    padding: 10px; /* 增加内边距，防止内容紧贴边框 */
  }
}

/* 专家分析卡片样式 */
.expert-card {
  background: #f8fafc;
  border-radius: 12px;
  margin-top: 0; /* 减少顶部边距 */
  margin-bottom: 0; /* 减少底部边距 */
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.expert-content {
  /* padding: 16px; */
  padding: 5px; /* 减少内边距 */
}

.analysis-item {
  margin-bottom: 20px; /* 减少底部边距 */
  background: white;
  padding: 8px; /* 减少内边距 */
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.analysis-item h3 {
  color: #2d8cf0;
  margin-bottom: 8px; /* 减少底部边距 */
  display: flex;
  align-items: center;
  gap: 8px;
}

.analysis-item p {
  line-height: 1.4; /* 适当减少行高 */
  color: #606266;
  white-space: pre-wrap;
}

</style>