
<template>
  <el-row :gutter="160" class="panel-group">
    <el-col :xs="16" :sm="16" :lg="24" class="card-panel-col">
      <div class="card-panel">
        <div class="card-panel-content">
          <div class="card-panel-icon-wrapper icon-mask">
            <svg-icon icon-class="eye-title" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              眼底图像处理系统
            </div>
          </div>
        </div>
      </div>
    </el-col>
  </el-row>
</template>

<script>
export default {
  props: {
    username: {
      type: String,
      default: ''
    }
  }

}
</script>
<style lang="scss" scoped>
.panel-group {
  margin-top: 18px;

  .card-panel-col {
    margin-bottom: 32px;
  }

  .card-panel {
    height: 108px;
    cursor: pointer;
    font-size: 12px;
    position: relative;
    overflow: hidden;
    color: #666;
    background: #fff;
    box-shadow: 4px 4px 40px rgba(0, 0, 0, .05);
    border-color: rgba(0, 0, 0, .05);

    &:hover {
      .card-panel-icon-wrapper {
        color: #fff;
      }

      .icon-mask {
        background: #304156;
      }

      .rectangle{
        width: 1000px;
      }
    }

    .card-panel-icon-wrapper {
      // 修改这一行
      margin: 0 16px 0 0; 
      padding: 16px;
      transition: all 0.38s ease-out;
      border-radius: 6px;
    }

    .card-panel-icon {
      float: left;
      font-size: 48px;
    }

    .card-panel-description {
      // 移除浮动
      float: none; 
      // 使用flex布局
      display: flex; 
      // 水平居中
      justify-content: center; 
      // 垂直居中
      align-items: center; 
      // 设置高度为100%以占满父元素高度
      height: 100%; 
      font-weight: bold;
      // 去除不必要的外边距
      margin: 0; 

      .card-panel-text {
        color: rgba(0, 0, 0, 0.5);
        font-size: 36px;
        letter-spacing: 10px;
      }

      .card-panel-num {
        font-size: 20px;
      }
    }

        .username-welcom{

      font-size: 36px;
      float: right;

    }

    // 新增
      .card-panel-content {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
  }

  }
}
</style>
