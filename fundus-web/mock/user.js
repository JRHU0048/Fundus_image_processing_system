// 定义一个对象 tokens，用于存储不同用户角色对应的 token
const tokens = {
  // 管理员用户对应的 token
  admin: {
      token: 'admin-token'
  },
  // 编辑人员用户对应的 token
  editor: {
      token: 'editor-token'
  }
};

// 定义一个对象 users，用于存储不同 token 对应的用户信息
const users = {
  // 'admin-token' 对应的用户信息
  'admin-token': {
      // 用户角色为管理员
      roles: ['admin'],
      // 用户简介
      introduction: 'I am a super administrator',
      // 用户头像链接
      avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
      // 用户姓名
      name: 'Super Admin'
  },
  // 'editor-token' 对应的用户信息
  'editor-token': {
      // 用户角色为编辑人员
      roles: ['editor'],
      // 用户简介
      introduction: 'I am an editor',
      // 用户头像链接
      avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
      // 用户姓名
      name: 'Normal Editor'
  }
};

// 导出一个数组，数组中的每个元素代表一个模拟的 API 接口
module.exports = [
  // 模拟用户登录接口
  {
      // 接口的 URL
      url: '/vue-admin-template/user/login',
      // 接口的请求方法为 POST
      type: 'post',
      // 接口的响应处理函数，接收 config 参数（包含请求的配置信息）
      response: config => {
          // 从请求体中解构出用户名
          const { username } = config.body;
          // 根据用户名从 tokens 对象中获取对应的 token
          const token = tokens[username];

          // 模拟登录失败的情况，如果未找到对应的 token
          if (!token) {
              return {
                  // 错误码，表示账号和密码错误
                  code: 60204,
                  // 错误信息
                  message: 'Account and password are incorrect.'
              };
          }

          // 登录成功，返回状态码 20000 和对应的 token
          return {
              // 成功状态码
              code: 20000,
              // 返回的数据，即 token
              data: token
          };
      }
  },

  // 模拟获取用户信息接口
  {
      // 接口的 URL，使用正则表达式匹配以该字符串开头的 URL
      url: '/vue-admin-template/user/info\.*',
      // 接口的请求方法为 GET
      type: 'get',
      // 接口的响应处理函数，接收 config 参数（包含请求的配置信息）
      response: config => {
          // 从请求的查询参数中解构出 token
          const { token } = config.query;
          // 根据 token 从 users 对象中获取对应的用户信息
          const info = users[token];

          // 模拟获取用户信息失败的情况，如果未找到对应的用户信息
          if (!info) {
              return {
                  // 错误码，表示登录失败，无法获取用户详情
                  code: 50008,
                  // 错误信息
                  message: 'Login failed, unable to get user details.'
              };
          }

          // 获取用户信息成功，返回状态码 20000 和对应的用户信息
          return {
              // 成功状态码
              code: 20000,
              // 返回的数据，即用户信息
              data: info
          };
      }
  },

  // 模拟用户登出接口
  {
      // 接口的 URL
      url: '/vue-admin-template/user/logout',
      // 接口的请求方法为 POST
      type: 'post',
      // 接口的响应处理函数，不使用传入的参数
      response: _ => {
          // 登出成功，返回状态码 20000 和成功信息
          return {
              // 成功状态码
              code: 20000,
              // 返回的数据，即成功信息
              data: 'success'
          };
      }
  }
];