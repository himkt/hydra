(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5482],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return l},kt:function(){return d}});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=r.createContext({}),u=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},l=function(e){var t=u(e.components);return r.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},f=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,c=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),f=u(n),d=o,g=f["".concat(c,".").concat(d)]||f[d]||p[d]||i;return n?r.createElement(g,a(a({ref:t},l),{},{components:n})):r.createElement(g,a({ref:t},l))}));function d(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,a=new Array(i);a[0]=f;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:o,a[1]=s;for(var u=2;u<i;u++)a[u]=n[u];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}f.displayName="MDXCreateElement"},9695:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return c},metadata:function(){return u},toc:function(){return l},default:function(){return f}});var r=n(2122),o=n(9756),i=(n(7294),n(3905)),a=["components"],s={id:"config_store",title:"Config Store API"},c=void 0,u={unversionedId:"tutorials/structured_config/config_store",id:"version-1.0/tutorials/structured_config/config_store",isDocsHomePage:!1,title:"Config Store API",description:"ConfigStore is a singleton storing configs in memory.",source:"@site/versioned_docs/version-1.0/tutorials/structured_config/10_config_store.md",sourceDirName:"tutorials/structured_config",slug:"/tutorials/structured_config/config_store",permalink:"/docs/1.0/tutorials/structured_config/config_store",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/versioned_docs/version-1.0/tutorials/structured_config/10_config_store.md",version:"1.0",lastUpdatedBy:"Jieru Hu",lastUpdatedAt:1629405728,formattedLastUpdatedAt:"8/19/2021",sidebarPosition:10,frontMatter:{id:"config_store",title:"Config Store API"},sidebar:"version-1.0/docs",previous:{title:"Dynamic schema with many configs",permalink:"/docs/1.0/tutorials/structured_config/dynamic_schema"},next:{title:"Overview",permalink:"/docs/1.0/patterns/instantiate_objects/overview"}},l=[{value:"API",id:"api",children:[]},{value:"Example node values",id:"example-node-values",children:[]}],p={toc:l};function f(e){var t=e.components,n=(0,o.Z)(e,a);return(0,i.kt)("wrapper",(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("p",null,(0,i.kt)("inlineCode",{parentName:"p"},"ConfigStore")," is a singleton storing configs in memory.\nThe primary API for interacting with the ",(0,i.kt)("inlineCode",{parentName:"p"},"ConfigStore")," is the store method described below."),(0,i.kt)("h3",{id:"api"},"API"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'class ConfigStore(metaclass=Singleton):\n    def store(\n        self,\n        name: str,\n        node: Any,\n        group: Optional[str] = None,\n        package: Optional[str] = "_group_",\n        provider: Optional[str] = None,\n    ) -> None:\n        """\n        Stores a config node into the repository\n        :param name: config name\n        :param node: config node, can be DictConfig, ListConfig,\n            Structured configs and even dict and list\n        :param group: config group, subgroup separator is \'/\',\n            for example hydra/launcher\n        :param package: Config node parent hierarchy.\n            Child separator is \'.\', for example foo.bar.baz\n        :param provider: the name of the module/app providing this config.\n            Helps debugging.\n        """\n    ...\n')),(0,i.kt)("h3",{id:"example-node-values"},"Example node values"),(0,i.kt)("p",null,"A few examples of supported node values parameters:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'@dataclass\nclass MySQLConfig:\n    host: str = "localhost"\n    port: int = 3306\n\n# Using the type\ncs.store(name="config1", node=MySQLConfig)\n# Using an instance, overriding some default values\ncs.store(name="config2", node=MySQLConfig(host="test.db", port=3307))\n# Using a dictionary, forfeiting runtime type safety\ncs.store(name="config3", node={"host": "localhost", "port": 3308})\n')))}f.isMDXComponent=!0}}]);