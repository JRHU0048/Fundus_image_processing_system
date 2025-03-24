<template>
  <div class="dashboard-container">

    <div class="introduction-text">
      <p>本模块专为眼底图像分割任务，能够提供精准的眼底图像视杯、视盘分割结果</p>
    </div>

    <div class="container-photo">
      <div class="left-option">
        <el-form>
          <el-form-item label="模型选择">
            <el-select v-model="modules" placeholder="请选择合适的模型">
              <el-option label="视杯分割模型" value="seg_cup" />
              <el-option label="视盘分割模型" value="seg_disc" />
              <el-option label="视杯、视盘分割模型" value="seg_cup_disc" />
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
            <div class="el-upload__text">请将<em>分割</em>图片拖到此处，或<em>点击上传</em></div>
            <div slot="tip" class="el-upload__tip">目前系统仅支持jpg/png格式的图片</div>
          </el-upload>
        </div>

        <!-- <img v-if="photoUrl!=null&&photoUrl!=''" :src="photoUrl" alt="" srcset=""> -->

        <!-- 左右布局 -->
        <div v-else style="display: flex; flex-direction: row;">
          <!-- 原始图片 -->
          <img :src="photoUrlBefore" alt="Original Image" style="width: 300px; height: 300px; margin-right: 10px;" />
          <!-- 处理后的图片 -->
          <img :src="photoUrl" alt="Processed Image" style="width: 300px; height: 300px;" />
        </div>
        <!-- 左右布局 -->

        <el-button v-if="photoUrl!=null&&photoUrl!=''" type="primary" style="width:200px;margin-top:30px" @click="onBack">重新上传</el-button>
        
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
      imageName: ''  // 存储上传图片的名称
      // modules: '',
      // uploadUrl: '',
      // videoForm: {
      //   Video: ''
      // },
      // videoFlag: true,
      // videoUploadPercent: 0,
      // videoName: ''
    }
  },
  methods: {
    onSubmit() {
      const that = this
      const loading = this.$loading({
        lock: true,
        text: '检测中...',
        spinner: 'el-icon-loading',
        background: 'rgba(0, 0, 0, 0.7)'
      })

      this.req({
        url: '/file/checkphoto',
        method: 'get',
        params: {
          model: that.modules,
          imageName: that.imageName
        }
      }).then((res) => {
        console.log(res)
        this.photoUrl = res.data.imageUrl
        loading.close()
      })
    },
    onBack() {
      this.photoUrl = ''
      this.photoUrlBefore = ''
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
    // beforeUploadVideo(file) {
    //   const isLt10M = file.size / 1024 / 1024 < 10
    //   if (
    //     [
    //       'video/mp4',
    //       'video/ogg',
    //       'video/flv',
    //       'video/avi',
    //       'video/wmv',
    //       'video/rmvb'
    //     ].indexOf(file.type) == -1
    //   ) {
    //     this.$message.error('请上传正确的视频格式')
    //     return false
    //   }
    //   if (!isLt10M) {
    //     this.$message.error('上传视频大小不能超过10MB哦!')
    //     return false
    //   }
    // },
    // uploadVideoProcess(event, file, fileList) {
    //   this.videoFlag = true
    //   console.log('percentage', file.percentage)
    //   this.videoUploadPercent = file.percentage.toFixed(0)
    // },
    // handleVideoSuccess(res, file) {
    //   this.videoUploadPercent = 100
    //   this.videoFlag = false
    //   console.log(res)
    //   this.videoForm['Video'] = res.videoUrl
    //   this.videoName = res.videoName
    // }
  }
}
</script>


<style>
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
  /* margin: 30px; */
  margin: 0 auto;
}

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
}
</style>
