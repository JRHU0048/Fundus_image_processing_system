// 导入 Vue 框架
import Vue from 'vue';
// 导入 Vue Router 插件，用于实现单页面应用的路由功能
import Router from 'vue-router';

// 使用 Vue.use() 方法安装 Vue Router 插件
Vue.use(Router);

/* Layout */
// 导入项目的布局组件，通常包含侧边栏、导航栏等公共部分
import Layout from '@/layout';

/**
 * 注意：只有当路由的子路由数量大于等于 1 时，子菜单才会显示
 * 详细信息请参考：https://panjiachen.github.io/vue-element-admin-site/guide/essentials/router-and-nav.html
 *
 * hidden: true                   如果设置为 true，该路由项将不会在侧边栏中显示（默认值为 false）
 * alwaysShow: true               如果设置为 true，将始终显示根菜单
 *                                如果未设置 alwaysShow，当路由项有多个子路由时，将变为嵌套模式，否则不显示根菜单
 * redirect: noRedirect           如果设置为 noRedirect，在面包屑导航中不会进行重定向
 * name:'router-name'             该名称用于 <keep-alive> 组件（必须设置！！！）
 * meta : {
    roles: ['admin','editor']    控制页面的访问角色（可以设置多个角色）
    title: 'title'               侧边栏和面包屑导航中显示的名称（建议设置）
    icon: 'svg-name'/'el-icon-x' 侧边栏中显示的图标
    breadcrumb: false            如果设置为 false，该路由项将在面包屑导航中隐藏（默认值为 true）
    activeMenu: '/example/list'  如果设置了路径，侧边栏将高亮显示你设置的路径
  }
 */

/**
 * constantRoutes
 * 不需要权限验证的基础页面
 * 所有角色都可以访问
 */
export const constantRoutes = [
  {
    // 根路径，重定向到登录页面
    path: '/',
    // 使用懒加载方式导入登录页面组件
    component: () => import('@/views/login/index'),
    // 该路由项不在侧边栏显示
    hidden: true
  },

  {
    // 404 页面路径
    path: '/404',
    // 懒加载 404 页面组件
    component: () => import('@/views/404'),
    // 该路由项不在侧边栏显示
    hidden: true
  },

  {
    // 注册页面路径
    path: '/register',
    // 懒加载注册页面组件
    component: () => import('@/views/register/index'),
    // 该路由项不在侧边栏显示
    hidden: true
  },

  {
    // 仪表盘页面路径
    path: '/dashboard',
    // 使用布局组件
    component: Layout,
    children: [
      {
        // 子路径为空，表示默认路径
        path: '',
        // 路由名称，用于 <keep-alive> 组件
        name: 'dashboard',
        // 懒加载仪表盘页面组件
        component: () => import('@/views/dashboard/index'),
        meta: {
          // 侧边栏和面包屑导航中显示的标题
          title: '首页',
          // 侧边栏中显示的图标
          // icon: 'dash-board'
          icon:'el-icon-orange'
        }
      }
    ]
  },

  {
    // 图片检测页面路径
    path: '/photo',
    // 使用布局组件
    component: Layout,
    children: [
      {
        // 子路径
        path: 'index',
        // 路由名称
        name: 'Form',
        // 懒加载图片检测页面组件
        component: () => import('@/views/photo/index'),
        meta: {
          // 侧边栏和面包屑导航中显示的标题
          // title: '图片检测',
          title: '眼底图像分类',
          // 侧边栏中显示的图标
          // icon: 'el-icon-s-help'
          icon:'el-icon-s-data'
        }
      }
    ]
  },

  {
    // 视频检测页面路径
    path: '/video',
    // 使用布局组件
    component: Layout,
    children: [
      {
        // 子路径
        path: 'index',
        // 路由名称
        name: 'Form',
        // 懒加载视频检测页面组件
        component: () => import('@/views/video/index'),
        meta: {
          // 侧边栏和面包屑导航中显示的标题
          // title: '视频检测',
          title: '眼底图像分割',
          // 侧边栏中显示的图标
          icon: 'el-icon-s-grid'
        }
      }
    ]
  },

  // {
  //   // 监控检测页面路径
  //   path: '/camera',
  //   // 使用布局组件
  //   component: Layout,
  //   children: [
  //     {
  //       // 子路径
  //       path: 'index',
  //       // 路由名称
  //       name: 'Form',
  //       // 懒加载监控检测页面组件
  //       component: () => import('@/views/camera/index'),
  //       meta: {
  //         // 侧边栏和面包屑导航中显示的标题
  //         title: '监控检测',
  //         // 侧边栏中显示的图标
  //         icon: 'camera'
  //       }
  //     }
  //   ]
  // },

  {
    // 个人中心页面路径
    path: '/nested',
    // 使用布局组件
    component: Layout,
    // 重定向到子路由 /nested/menu1
    redirect: '/nested/menu1',
    // 路由名称
    name: 'Nested',
    meta: {
      // 侧边栏和面包屑导航中显示的标题
      title: '个人中心',
      // 侧边栏中显示的图标
      // icon: 'nested'
      icon:'el-icon-s-custom'
    },
    children: [
      {
        // 子路径
        path: 'menu1',
        // 懒加载用户信息页面组件
        component: () => import('@/views/nested/menu1/index'),
        // 路由名称
        name: 'Menu1',
        meta: {
          // 侧边栏和面包屑导航中显示的标题
          title: '用户信息'
        }
      },
      {
        // 子路径
        path: 'menu2',
        // 懒加载修改密码页面组件
        component: () => import('@/views/nested/menu2/index'),
        // 路由名称
        name: 'Menu2',
        meta: {
          // 侧边栏和面包屑导航中显示的标题
          title: '修改密码'
        }
      },
      {
        // 子路径，这里 '../' 可能是返回上一级的操作
        path: '../',
        // 路由名称
        name: 'Menu2',
        meta: {
          // 侧边栏和面包屑导航中显示的标题
          title: '退出登录'
        }
      }
    ]
  },

  // {
  //   path: 'external-link',
  //   component: Layout,
  //   children: [
  //     {
  //       path: 'https://panjiachen.github.io/vue-element-admin-site/#/',
  //       meta: { title: 'External Link', icon: 'link' }
  //     }
  //   ]
  // },

  {
    // 系统开发介绍
    path: '/about',
    // 使用布局组件
    component: Layout,
    children: [
      {
        // 子路径
        path: 'index',
        // 路由名称
        name: 'ProjectIntroduction',
        // 懒加载视频检测页面组件
        component: () => import('@/views/about/index'),
        meta: {
          // 侧边栏和面包屑导航中显示的标题
          // title: '视频检测',
          title: '系统开发介绍',
          // 侧边栏中显示的图标
          icon: 'el-icon-view'
        }
      }
    ]
  },

  // 404 页面必须放在最后！！！
  {
    // 匹配所有未定义的路径
    path: '*',
    // 重定向到 404 页面
    redirect: '/404',
    // 该路由项不在侧边栏显示
    hidden: true
  }
];

// 创建一个新的 Vue Router 实例的函数
const createRouter = () => new Router({
  // 路由模式，这里注释掉了 'history' 模式，使用该模式需要服务器支持
  // mode: 'history', 
  // 滚动行为，每次路由切换后将页面滚动到顶部
  scrollBehavior: () => ({
    y: 0
  }),
  // 路由配置，使用 constantRoutes 数组
  routes: constantRoutes
});

// 创建一个 Vue Router 实例
const router = createRouter();

// 详细信息请参考：https://github.com/vuejs/vue-router/issues/1234#issuecomment-357941465
// 重置路由的函数，用于在用户权限变化等情况下重置路由配置
export function resetRouter() {
  // 创建一个新的路由实例
  const newRouter = createRouter();
  // 重置当前路由实例的匹配器
  router.matcher = newRouter.matcher; 
}

// 导出默认的路由实例
export default router;