<template>
  <div class="panel-group-with-background">
    <div class="function-introduction-container">
      <p class="function-introduction">本系统具备强大的眼底图像处理功能，能够对眼底图像进行精准的分类和分割任务，为医疗机构提供高效、准确的检测结果。</p>
    </div>
    <el-row :gutter="160" class="panel-group">
      <el-col :xs="16" :sm="16" :lg="8" class="card-panel-col">
        <div class="card-panel" @click="handleSetLineChartData('newVisitis')">
          <div class="card-panel-icon-wrapper icon-people">
            <svg-icon icon-class="cpu" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              CPU占用率
            </div>
            <count-to :start-val="0" :end-val="cpu_percent" :duration="2600" class="card-panel-num" />%
          </div>
        </div>
      </el-col>
      <el-col :xs="16" :sm="16" :lg="8" class="card-panel-col">
        <div class="card-panel" @click="handleSetLineChartData('messages')">
          <div class="card-panel-icon-wrapper icon-message">
            <svg-icon icon-class="memory" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              内存使用率
            </div>
            <count-to :start-val="0" :end-val="memory_percent" :duration="3000" class="card-panel-num" />%
          </div>
        </div>
      </el-col>
      <el-col :xs="16" :sm="16" :lg="8" class="card-panel-col">
        <div class="card-panel" @click="handleSetLineChartData('purchases')">
          <div class="card-panel-icon-wrapper icon-money">
            <svg-icon icon-class="network" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              带宽
            </div>
            <count-to :start-val="0" :end-val="delta" :duration="3200" class="card-panel-num" />Mbps
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import CountTo from 'vue-count-to';

export default {
  components: {
    CountTo
  },
  data() {
    return {
      cpu_percent: 4,
      memory_percent: 12,
      delta: 20
    };
  },
  mounted() {
    let that = this;
    setInterval(() => {
      this.req({
        url: '/util/getComInfo',
        method: 'get'
      }).then(res => {
        console.log(res);
        that.cpu_percent = res.data.cpu_percent;
        that.memory_percent = res.data.memory_percent;
        that.delta = navigator.connection.downlink;
      });
    }, 10000);
  },
  methods: {
    handleSetLineChartData(type) {
      this.$emit('handleSetLineChartData', type);
    }
  }
};
</script>

<style lang="scss" scoped>
.panel-group-with-background {
  position: relative;
  min-height: 65vh; 
  margin-top: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  &::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: url('https://pic1.imgdb.cn/item/67d539f788c538a9b5beb21d.png');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    // border-radius: 8px; /* 圆角 */
    filter: blur(1px);
    z-index: 1;
  }
}

.function-introduction-container {
  position: relative;
  z-index: 2;
  width: 94%;
  padding: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.function-introduction {
  font-size: 25px;
  font-weight: bold;
  line-height: 1.6;
  color: rgba(0, 0, 0, 0.7);
  text-align: center;
  background-color: rgba(255, 255, 255, 0.4);
  backdrop-filter: blur(2px);
  border-radius: 8px;
  padding: 20px;
  width: 100%;
}

.panel-group {
  margin-top: 90px;
  position: relative;
  z-index: 2;
  width: 105%;
  padding: 0px;

  .card-panel-col {
    margin-bottom: 0px;
  }

  .card-panel {
    height: 108px;
    width: 100%;
    cursor: pointer; /* 鼠标悬停时显示手势 */
    font-size: 14px;
    position: center;
    // overflow: hidden; /* 溢出部分隐藏 */
    color: rgba(0, 0, 0, 0.7); /* 字体颜色 */
    background-color: rgba(255, 255, 255, 0.4); /* 背景颜色 */
    backdrop-filter: blur(2px); /* 模糊效果 */
    border-radius: 8px; /* 圆角 */

    &:hover {
      .card-panel-icon-wrapper {
        color: #fff;
      }

      .icon-people {
        background: #40c9c6;
      }

      .icon-message {
        background: #36a3f7;
      }

      .icon-money {
        background: #f4516c;
      }

      .icon-shopping {
        background: #34bfa3;
      }
    }

    .icon-people {
      color: #40c9c6;
    }

    .icon-message {
      color: #36a3f7;
    }

    .icon-money {
      color: #f4516c;
    }

    .icon-shopping {
      color: #34bfa3;
    }

    .card-panel-icon-wrapper {
      float: left;
      margin: 14px 0 0 14px;
      padding: 14px;
      transition: all 0.38s ease-out;
      border-radius: 6px;
    }

    .card-panel-icon {
      float: left;
      font-size: 48px;
    }

    .card-panel-description {
      float: right;
      font-weight: bold;
      margin: 26px;
      margin-left: 0px;

      .card-panel-text {
        line-height: 18px;
        color: rgba(0, 0, 0, 0.7);
        font-size: 17px;
        margin-bottom: 12px;
      }

      .card-panel-num {
        font-size: 20px;
      }
    }
  }
}

@media (max-width: 550px) {
  .card-panel-description {
    display: none;
  }

  .card-panel-icon-wrapper {
    float: none !important;
    width: 100%;
    height: 100%;
    margin: 0 !important;

    .svg-icon {
      display: block;
      margin: 14px auto !important;
      float: none !important;
    }
  }
}
</style>  
