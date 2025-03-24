<template>
  <div class="dashboard-container">
    <WelcomPanel />
    <PanelGroup />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import PanelGroup from './components/panelGroup';
import WelcomPanel from './components/welcomPanel';

export default {
  name: 'Dashboard',
  components: {
    PanelGroup,
    WelcomPanel
  },
  computed: {
    ...mapGetters([
      'name'
    ])
  }
};
</script>

<style lang="scss" scoped>
.dashboard {
  &-container {
    margin: 30px;
  }
  &-text {
    font-size: 30px;
    line-height: 46px;
  }
}

.bottom-container {
  position: relative;
  margin-top: 30px;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0;
  height: 250px;

  &::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: url('https://pic1.imgdb.cn/item/67d4284488c538a9b5bdae33.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    filter: blur(1px);
    z-index: 1;
    border-radius: 8px;
  }

 .function-introduction {
    position: relative;
    z-index: 2;
    font-size: 25px;
    font-weight: bold;
    line-height: 1.6;
    color: rgba(0, 0, 0, 0.65);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 100%;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.3);
    backdrop-filter: blur(2px);
    border-radius: 8px;
    padding: 0 20px;
  }
}
</style>    